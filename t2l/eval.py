"""Evaluation, description ablation, and t-SNE visualization."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from t2l.data import (
    build_description_bank,
    dezscore_lora,
    example_to_prompt,
    load_description_bank,
    load_hf_benchmark,
    load_lora_dict,
)
from t2l.model import (
    HyperModulator,
    ModuleSpec,
    TaskEmbedder,
    attach_dynamic_lora,
    clear_lora,
    inject_generated_lora,
    inspect_target_modules,
)
from t2l.train import get_device, load_base, set_seed


def _load_hyper_ckpt(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    specs = [ModuleSpec(n, i, o) for n, i, o in ckpt["module_specs"]]
    use_text = ckpt.get("use_text", True)
    n_tasks = len(ckpt.get("tasks") or ckpt.get("train_tasks") or [0])
    hyper = HyperModulator(
        variant=cfg["variant"],
        n_layers=ckpt["n_layers"],
        module_specs=specs,
        rank=cfg["rank"],
        task_in_dim=cfg.get("embedder_dim", 1024),
        task_emb_dim=cfg["task_emb_dim"],
        module_emb_dim=cfg["module_emb_dim"],
        layer_emb_dim=cfg["layer_emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        mlp_expand=cfg["mlp_expand"],
        dropout=cfg["dropout"],
        use_text_task=use_text,
        n_tasks=n_tasks,
    )
    hyper.load_state_dict(ckpt["hyper"])
    hyper.to(device).eval()
    return hyper, ckpt, use_text


@torch.no_grad()
def generate_answer(model, tok, prompt: str, max_new: int = 16) -> str:
    device = next(model.parameters()).device
    ids = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **ids,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    gen = out[0, ids["input_ids"].size(1) :]
    return tok.decode(gen, skip_special_tokens=True).strip()


def normalize_pred(task: str, text: str) -> str:
    t = text.strip().split()[0] if text.strip() else ""
    t = t.strip(".,:;\"'`").lower()
    if task in {"boolq"}:
        if "true" in text.lower():
            return "true"
        if "false" in text.lower():
            return "false"
    if task in {"arc_easy", "arc_challenge", "openbookqa"}:
        for ch in text.upper():
            if ch in "ABCD":
                return ch
    if task in {"piqa", "hellaswag"}:
        for ch in text:
            if ch in "0123":
                return ch
    if task == "winogrande":
        for ch in text:
            if ch in "12":
                return ch
    return t


@torch.no_grad()
def eval_model_on_task(
    model,
    tok,
    task: str,
    max_samples: int = 200,
    split: str = "validation",
) -> float:
    try:
        ds = load_hf_benchmark(task, split=split, max_samples=max_samples)
    except Exception:
        ds = load_hf_benchmark(task, split="test", max_samples=max_samples)
    correct, total = 0, 0
    for ex in tqdm(ds, desc=f"eval:{task}", leave=False):
        try:
            prompt, gold = example_to_prompt(task, ex)
        except Exception:
            continue
        pred = normalize_pred(task, generate_answer(model, tok, prompt))
        gold_n = normalize_pred(task, str(gold))
        correct += int(pred == gold_n)
        total += 1
    return correct / max(total, 1)


def eval_base(cfg: Dict[str, Any]) -> Dict[str, float]:
    device = get_device()
    model, tok, _ = load_base(cfg, device)
    model.eval()
    scores = {}
    for task in cfg["eval"]["benches"]:
        scores[task] = eval_model_on_task(
            model, tok, task, max_samples=cfg["eval"]["max_samples"]
        )
    scores["avg"] = sum(scores.values()) / len(scores)
    return scores


def eval_oracle_loras(cfg: Dict[str, Any]) -> Dict[str, float]:
    from peft import PeftModel

    device = get_device()
    scores = {}
    for task in cfg["oracle"]["tasks"]:
        if task not in cfg["eval"]["benches"]:
            continue
        base, tok, _ = load_base(cfg, device)
        path = Path(cfg["output_dir"]) / "oracles" / task
        if not path.exists():
            continue
        model = PeftModel.from_pretrained(base, str(path))
        model.eval()
        scores[task] = eval_model_on_task(
            model, tok, task, max_samples=cfg["eval"]["max_samples"]
        )
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if scores:
        scores["avg"] = sum(scores.values()) / len(scores)
    return scores


def eval_recon_hyper(cfg: Dict[str, Any]) -> Dict[str, float]:
    device = get_device()
    ckpt_path = Path(cfg["output_dir"]) / "recon" / "best.pt"
    if not ckpt_path.exists():
        return {}
    hyper, ckpt, use_text = _load_hyper_ckpt(ckpt_path, device)
    stats = torch.load(
        Path(cfg["output_dir"]) / "recon" / "zscore_stats.pt",
        map_location="cpu",
        weights_only=False,
    )
    base, tok, _ = load_base(cfg, device)
    lookup = attach_dynamic_lora(
        base, cfg["target_modules"], cfg["rank"], cfg["lora_alpha"]
    )
    base.eval()
    embedder = TaskEmbedder(cfg["embedder_model"]).to(device) if use_text else None
    desc_path = Path("data/bench_descriptions.json")
    if not desc_path.exists():
        build_description_bank(ckpt["tasks"], 8, desc_path)
    bank = load_description_bank(desc_path)

    scores = {}
    for task in cfg["eval"]["benches"]:
        if task not in ckpt["tasks"]:
            continue
        if use_text:
            desc = bank[task][0]
            te = embedder([desc], device).squeeze(0)
        else:
            te = torch.tensor(ckpt["tasks"].index(task), device=device)
        with torch.no_grad():
            lora = hyper.generate_lora(te)
            lora = dezscore_lora(lora, stats)
            # move to device
            lora = {
                m: {k: v.to(device) for k, v in mats.items()}
                for m, mats in lora.items()
            }
            inject_generated_lora(lookup, lora)
        scores[task] = eval_model_on_task(
            base, tok, task, max_samples=cfg["eval"]["max_samples"]
        )
    clear_lora(lookup)
    if scores:
        scores["avg"] = sum(scores.values()) / len(scores)
    return scores


def eval_sft_zeroshot(cfg: Dict[str, Any]) -> Dict[str, float]:
    device = get_device()
    ckpt_path = Path(cfg["output_dir"]) / "sft" / "final.pt"
    if not ckpt_path.exists():
        return {}
    hyper, ckpt, _ = _load_hyper_ckpt(ckpt_path, device)
    base, tok, _ = load_base(cfg, device)
    lookup = attach_dynamic_lora(
        base, cfg["target_modules"], cfg["rank"], cfg["lora_alpha"]
    )
    base.eval()
    embedder = TaskEmbedder(cfg["embedder_model"]).to(device)
    desc_path = Path("data/bench_descriptions.json")
    if not desc_path.exists():
        build_description_bank(cfg["eval"]["benches"], 8, desc_path)
    bank = load_description_bank(desc_path)

    scores = {}
    for task in cfg["eval"]["benches"]:
        desc = bank.get(task, [f"Solve {task} questions."])[0]
        te = embedder([desc], device).squeeze(0)
        with torch.no_grad():
            lora = hyper.generate_lora(te)
            inject_generated_lora(lookup, lora)
        scores[task] = eval_model_on_task(
            base, tok, task, max_samples=cfg["eval"]["max_samples"]
        )
    clear_lora(lookup)
    if scores:
        scores["avg"] = sum(scores.values()) / len(scores)
    return scores


def ablation_descriptions(cfg: Dict[str, Any]) -> Dict[str, float]:
    """Aligned vs random description (Table 5 style) using recon hyper."""
    device = get_device()
    ckpt_path = Path(cfg["output_dir"]) / "recon" / "best.pt"
    if not ckpt_path.exists():
        return {}
    hyper, ckpt, use_text = _load_hyper_ckpt(ckpt_path, device)
    if not use_text:
        return {}
    stats = torch.load(
        Path(cfg["output_dir"]) / "recon" / "zscore_stats.pt",
        map_location="cpu",
        weights_only=False,
    )
    base, tok, _ = load_base(cfg, device)
    lookup = attach_dynamic_lora(
        base, cfg["target_modules"], cfg["rank"], cfg["lora_alpha"]
    )
    embedder = TaskEmbedder(cfg["embedder_model"]).to(device)
    bank = load_description_bank(Path("data/bench_descriptions.json"))
    task = cfg["eval"]["benches"][0]

    results = {}
    for label, desc in [
        ("aligned", bank[task][0]),
        ("random_string", "xqz plork 99 blue banana timing belt"),
        (
            "other_task",
            bank[[t for t in bank if t != task][0]][0]
            if len(bank) > 1
            else "unrelated task",
        ),
    ]:
        te = embedder([desc], device).squeeze(0)
        with torch.no_grad():
            lora = dezscore_lora(hyper.generate_lora(te), stats)
            lora = {
                m: {k: v.to(device) for k, v in mats.items()}
                for m, mats in lora.items()
            }
            inject_generated_lora(lookup, lora)
        results[label] = eval_model_on_task(
            base, tok, task, max_samples=min(100, cfg["eval"]["max_samples"])
        )
    return results


def viz_tsne(cfg: Dict[str, Any]) -> Optional[Path]:
    """t-SNE of task-encoder outputs for benchmark descriptions (Fig 5 style)."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("matplotlib/sklearn required for viz")
        return None

    device = get_device()
    ckpt_path = Path(cfg["output_dir"]) / "sft" / "final.pt"
    if not ckpt_path.exists():
        ckpt_path = Path(cfg["output_dir"]) / "recon" / "best.pt"
    if not ckpt_path.exists():
        return None

    hyper, ckpt, use_text = _load_hyper_ckpt(ckpt_path, device)
    if not use_text:
        return None
    embedder = TaskEmbedder(cfg["embedder_model"]).to(device)
    desc_path = Path("data/bench_descriptions.json")
    if not desc_path.exists():
        build_description_bank(cfg["eval"]["benches"], 8, desc_path)
    bank = load_description_bank(desc_path)

    vecs, labels = [], []
    with torch.no_grad():
        for task, descs in bank.items():
            for d in descs[:3]:
                te = embedder([d], device)
                enc = hyper.encode_task(te).cpu().numpy()[0]
                vecs.append(enc)
                labels.append(task)

    import numpy as np

    X = np.stack(vecs)
    if len(X) < 4:
        return None
    perp = max(2, min(5, len(X) - 1))
    xy = TSNE(n_components=2, perplexity=perp, random_state=0).fit_transform(X)

    plt.figure(figsize=(7, 5))
    for task in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == task]
        plt.scatter(xy[idx, 0], xy[idx, 1], label=task, s=40)
    plt.legend(fontsize=8, loc="best")
    plt.title("T2L task-encoder activations (t-SNE)")
    plt.tight_layout()
    out = Path(cfg["output_dir"]) / "tsne_task_encoder.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")
    return out


def run_eval(cfg: Dict[str, Any]) -> Dict[str, Any]:
    set_seed(cfg["seed"])
    out: Dict[str, Any] = {}
    print("Evaluating base model...")
    out["base"] = eval_base(cfg)
    print(out["base"])

    print("Evaluating oracle LoRAs...")
    out["oracle"] = eval_oracle_loras(cfg)
    print(out["oracle"])

    print("Evaluating recon T2L...")
    out["recon_t2l"] = eval_recon_hyper(cfg)
    print(out["recon_t2l"])

    print("Evaluating SFT T2L zero-shot...")
    out["sft_t2l"] = eval_sft_zeroshot(cfg)
    print(out["sft_t2l"])

    print("Description ablation...")
    out["ablation"] = ablation_descriptions(cfg)
    print(out["ablation"])

    viz_tsne(cfg)

    path = Path(cfg["output_dir"]) / "eval_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {path}")
    return out
