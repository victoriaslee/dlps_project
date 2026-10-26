"""Oracle LoRA training, reconstruction training, and hypernetwork SFT."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from t2l.data import (
    BenchmarkSFTDataset,
    HyperSFTDataset,
    build_description_bank,
    compute_zscore_stats,
    dezscore_lora,
    load_description_bank,
    load_lora_dict,
    peft_state_to_lora_dict,
    prepare_sni_split,
    save_lora_dict,
    zscore_lora,
)
from t2l.model import (
    HyperModulator,
    TaskEmbedder,
    attach_dynamic_lora,
    clear_lora,
    inject_generated_lora,
    inspect_target_modules,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon — much faster than CPU for this workload
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_base(cfg: Dict[str, Any], device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # dtype: bf16/fp16 on CUDA; fp16 on MPS; fp32 on CPU (bf16 on CPU is very slow)
    if device.type == "cuda":
        dtype = torch.bfloat16 if cfg.get("dtype") == "bfloat16" else torch.float16
    elif device.type == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
        print(
            "WARNING: running on CPU — expect very slow steps. "
            "Use a CUDA GPU or Apple MPS, or keep config.yaml lite settings."
        )

    print(f"Loading {cfg['base_model']} on {device} ({dtype})")
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    # Gradient checkpointing hurts a lot on CPU and is unnecessary at lite batch sizes
    if device.type == "cuda" and cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to(device)
    return model, tok, dtype


# stage 1: oracle LoRA training

def train_oracles(cfg: Dict[str, Any]) -> None:
    from peft import LoraConfig, TaskType, get_peft_model

    set_seed(cfg["seed"])
    device = get_device()
    out_dir = Path(cfg["output_dir"]) / "oracles"
    out_dir.mkdir(parents=True, exist_ok=True)
    oc = cfg["oracle"]

    for task in oc["tasks"]:
        print(f"\n=== Oracle LoRA: {task} ===")
        model, tok, dtype = load_base(cfg, device)
        # Inspect before PEFT wraps the linears
        n_layers, specs = inspect_target_modules(model, cfg["target_modules"])
        lora_cfg = LoraConfig(
            r=cfg["rank"],
            lora_alpha=cfg["lora_alpha"],
            target_modules=cfg["target_modules"],
            lora_dropout=cfg["dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_rslora=True,
        )
        model = get_peft_model(model, lora_cfg)
        model.train()
        ds = BenchmarkSFTDataset(
            task,
            tok,
            max_seq_len=oc["max_seq_len"],
            max_samples=oc["max_train_samples"],
        )
        loader = DataLoader(ds, batch_size=oc["batch_size"], shuffle=True)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=oc["lr"]
        )
        step, accum = 0, oc["grad_accum"]
        opt.zero_grad()
        pbar = tqdm(total=oc["max_steps"], desc=task)
        while step < oc["max_steps"]:
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss / accum
                loss.backward()
                if (step + 1) % accum == 0:
                    opt.step()
                    opt.zero_grad()
                step += 1
                pbar.set_postfix(loss=float(loss) * accum)
                pbar.update(1)
                if step >= oc["max_steps"]:
                    break
        pbar.close()

        # save PEFT + packed dict
        save_path = out_dir / task
        model.save_pretrained(save_path)
        tok.save_pretrained(save_path)
        state = {
            k: v
            for k, v in model.state_dict().items()
            if "lora_A" in k or "lora_B" in k
        }
        packed = peft_state_to_lora_dict(
            state, [s.name for s in specs], n_layers
        )
        save_lora_dict(out_dir / f"{task}.pt", packed)
        # free
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    meta = {"tasks": oc["tasks"], "rank": cfg["rank"], "modules": cfg["target_modules"]}
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Oracles saved to {out_dir}")


#stage 2: reconstruction hypernetwork

def _build_hyper(
    cfg: Dict[str, Any],
    n_layers: int,
    module_specs,
    n_tasks: int,
    use_text: bool,
) -> HyperModulator:
    return HyperModulator(
        variant=cfg["variant"],
        n_layers=n_layers,
        module_specs=module_specs,
        rank=cfg["rank"],
        task_in_dim=cfg["embedder_dim"],
        task_emb_dim=cfg["task_emb_dim"],
        module_emb_dim=cfg["module_emb_dim"],
        layer_emb_dim=cfg["layer_emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        mlp_expand=cfg["mlp_expand"],
        dropout=cfg["dropout"],
        use_text_task=use_text,
        n_tasks=n_tasks,
    )


def train_recon(cfg: Dict[str, Any]) -> None:
    set_seed(cfg["seed"])
    device = get_device()
    rc = cfg["recon"]
    out_dir = Path(cfg["output_dir"]) / "recon"
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir = Path(cfg["output_dir"]) / "oracles"

    # load library
    with open(oracle_dir / "meta.json") as f:
        meta = json.load(f)
    tasks = meta["tasks"]
    library = {t: load_lora_dict(oracle_dir / f"{t}.pt") for t in tasks}
    stats = compute_zscore_stats(library)
    torch.save(stats, out_dir / "zscore_stats.pt")

    # probe dims from first adapter
    first = next(iter(library.values()))
    n_layers = next(iter(first.values()))["A"].shape[0]
    from t2l.model import ModuleSpec

    module_specs = []
    for name in cfg["target_modules"]:
        A = first[name]["A"]
        B = first[name]["B"]
        module_specs.append(
            ModuleSpec(name, in_features=A.shape[-1], out_features=B.shape[-2])
        )

    use_text = rc["conditioning"] != "onehot"
    embedder = None
    desc_bank = None
    if use_text:
        embedder = TaskEmbedder(cfg["embedder_model"]).to(device)
        cfg = {**cfg, "embedder_dim": embedder.out_dim}
        desc_path = Path("data/bench_descriptions.json")
        if not desc_path.exists():
            build_description_bank(tasks, rc["descriptions_per_task"], desc_path)
        desc_bank = load_description_bank(desc_path)

    hyper = _build_hyper(cfg, n_layers, module_specs, len(tasks), use_text).to(device)
    opt = torch.optim.AdamW(hyper.parameters(), lr=rc["lr"])

    task_ids = {t: i for i, t in enumerate(tasks)}
    z_targets = {t: zscore_lora(library[t], stats) for t in tasks}

    best = float("inf")
    pbar = tqdm(range(rc["max_steps"]), desc="recon")
    for step in pbar:
        batch_tasks = random.choices(tasks, k=rc["batch_size"])
        loss = 0.0
        for t in batch_tasks:
            if use_text:
                desc = random.choice(desc_bank[t])
                te = embedder([desc], device).squeeze(0)
            else:
                te = torch.tensor(task_ids[t], device=device)
            pred = hyper.generate_lora(te)
            tgt = z_targets[t]
            for mod in cfg["target_modules"]:
                loss = loss + F.l1_loss(pred[mod]["A"].float(), tgt[mod]["A"].to(device))
                loss = loss + F.l1_loss(pred[mod]["B"].float(), tgt[mod]["B"].to(device))
        loss = loss / len(batch_tasks)
        opt.zero_grad()
        loss.backward()
        opt.step()
        val = float(loss)
        pbar.set_postfix(l1=val)
        if val < best:
            best = val
            torch.save(
                {
                    "hyper": hyper.state_dict(),
                    "cfg": cfg,
                    "tasks": tasks,
                    "n_layers": n_layers,
                    "module_specs": [
                        (s.name, s.in_features, s.out_features) for s in module_specs
                    ],
                    "use_text": use_text,
                },
                out_dir / "best.pt",
            )
        if val < rc["early_stop_l1"]:
            print(f"Early stop at step {step}, L1={val:.6f}")
            break

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"best_l1": best, "conditioning": rc["conditioning"]}, f, indent=2)
    print(f"Reconstruction done. best L1={best:.6f} → {out_dir}")


# Stage 3: hypernetwork SFT

def train_sft(cfg: Dict[str, Any]) -> None:
    set_seed(cfg["seed"])
    device = get_device()
    sc = cfg["sft"]
    out_dir = Path(cfg["output_dir"]) / "sft"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data/sni")

    manifest = prepare_sni_split(
        sc["n_train_tasks"],
        sc["n_holdout_tasks"],
        sc["samples_per_task"],
        data_dir,
    )
    train_tasks = manifest["train_tasks"]
    desc_path = Path("data/sni_descriptions.json")
    if not desc_path.exists():
        build_description_bank(
            train_tasks + manifest["holdout_tasks"],
            sc["descriptions_per_task"],
            desc_path,
            sni_definitions=manifest.get("definitions"),
        )
    desc_bank = load_description_bank(desc_path)

    base, tok, dtype = load_base(cfg, device)
    for p in base.parameters():
        p.requires_grad = False
    base.eval()  # keep dropout off; grads still flow to hyper via LoRA path
    # for SFT we need train mode on wrappers but frozen base weights
    n_layers, module_specs = inspect_target_modules(base, cfg["target_modules"])
    lookup = attach_dynamic_lora(
        base, cfg["target_modules"], rank=cfg["rank"], alpha=cfg["lora_alpha"]
    )

    embedder = TaskEmbedder(cfg["embedder_model"]).to(device)
    cfg = {**cfg, "embedder_dim": embedder.out_dim}
    hyper = _build_hyper(
        cfg, n_layers, module_specs, len(train_tasks), use_text=True
    ).to(device)

    ds = HyperSFTDataset(
        train_tasks, data_dir, desc_bank, tok, max_seq_len=sc["max_seq_len"]
    )

    def collate(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "description": [b["description"] for b in batch],
            "task": [b["task"] for b in batch],
        }

    loader = DataLoader(
        ds, batch_size=sc["batch_size"], shuffle=True, collate_fn=collate
    )

    opt = torch.optim.AdamW(hyper.parameters(), lr=sc["lr"])
    accum = sc["grad_accum"]
    step = 0
    opt.zero_grad()
    pbar = tqdm(total=sc["max_steps"], desc="sft")
    running = 0.0

    while step < sc["max_steps"]:
        for batch in loader:
            descs = batch["description"]
            if isinstance(descs, str):
                descs = [descs]
            # DataLoader collates strings into lists
            te = embedder(list(descs), device)
            # generate LoRA for first item if batch>1 is used sequentially (memory)
            losses = []
            for b in range(te.size(0)):
                lora = hyper.generate_lora(te[b])
                inject_generated_lora(lookup, lora)
                out = base(
                    input_ids=batch["input_ids"][b : b + 1].to(device),
                    attention_mask=batch["attention_mask"][b : b + 1].to(device),
                    labels=batch["labels"][b : b + 1].to(device),
                )
                losses.append(out.loss)
            loss = torch.stack(losses).mean() / accum
            loss.backward()
            running += float(loss) * accum
            if (step + 1) % accum == 0:
                opt.step()
                opt.zero_grad()
            step += 1
            pbar.update(1)
            if step % 50 == 0:
                pbar.set_postfix(loss=running / 50)
                running = 0.0
            if step % 2000 == 0:
                torch.save(
                    {
                        "hyper": hyper.state_dict(),
                        "cfg": cfg,
                        "train_tasks": train_tasks,
                        "n_layers": n_layers,
                        "module_specs": [
                            (s.name, s.in_features, s.out_features)
                            for s in module_specs
                        ],
                    },
                    out_dir / f"step_{step}.pt",
                )
            if step >= sc["max_steps"]:
                break

    pbar.close()
    torch.save(
        {
            "hyper": hyper.state_dict(),
            "cfg": cfg,
            "train_tasks": train_tasks,
            "holdout_tasks": manifest["holdout_tasks"],
            "n_layers": n_layers,
            "module_specs": [
                (s.name, s.in_features, s.out_features) for s in module_specs
            ],
        },
        out_dir / "final.pt",
    )
    clear_lora(lookup)
    print(f"SFT complete → {out_dir / 'final.pt'}")


def train_multitask_lora(cfg: Dict[str, Any]) -> None:
    """Baseline: one LoRA on union of oracle tasks."""
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import ConcatDataset

    set_seed(cfg["seed"])
    device = get_device()
    out_dir = Path(cfg["output_dir"]) / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    oc = cfg["oracle"]

    model, tok, dtype = load_base(cfg, device)
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg["rank"],
            lora_alpha=cfg["lora_alpha"],
            target_modules=cfg["target_modules"],
            lora_dropout=cfg["dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_rslora=True,
        ),
    )
    sets = [
        BenchmarkSFTDataset(t, tok, oc["max_seq_len"], oc["max_train_samples"] // 2)
        for t in oc["tasks"]
    ]
    ds = ConcatDataset(sets)
    loader = DataLoader(ds, batch_size=oc["batch_size"], shuffle=True)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=oc["lr"]
    )
    steps = min(oc["max_steps"] * 2, 2000)
    step, accum = 0, oc["grad_accum"]
    opt.zero_grad()
    pbar = tqdm(total=steps, desc="multitask-lora")
    while step < steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            (model(**batch).loss / accum).backward()
            if (step + 1) % accum == 0:
                opt.step()
                opt.zero_grad()
            step += 1
            pbar.update(1)
            if step >= steps:
                break
    pbar.close()
    model.save_pretrained(out_dir / "multitask_lora")
    print(f"Multi-task LoRA → {out_dir / 'multitask_lora'}")
