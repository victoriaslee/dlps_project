"""Hypernetwork (L/M/S), task embedder, and LoRA injection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

#building blocks
class ResidualMLP(nn.Module):
    def __init__(self, dim: int, expand: int, dropout: float = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, expand)
        self.fc2 = nn.Linear(expand, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.drop(F.silu(self.fc1(h)))
        h = self.drop(F.silu(self.fc2(h)))
        return x + h

@dataclass
class ModuleSpec:
    name: str
    in_features: int
    out_features: int

def _as_linear(mod: nn.Module) -> Optional[nn.Linear]:
    """Return an nn.Linear, unwrapping PEFT LoraLinear.base_layer if needed."""
    if isinstance(mod, nn.Linear):
        return mod
    base = getattr(mod, "base_layer", None)
    if isinstance(base, nn.Linear):
        return base
    # Some PEFT versions expose in/out on the wrapper itself
    if hasattr(mod, "in_features") and hasattr(mod, "out_features"):
        return mod  # type: ignore[return-value]
    return None


def inspect_target_modules(
    base_model: nn.Module, target_names: List[str]
) -> Tuple[int, List[ModuleSpec]]:
    """Infer #layers and per-module (in, out) from a HF causal LM (or PEFT model)."""
    specs: Dict[str, ModuleSpec] = {}
    n_layers = 0
    for name, mod in base_model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in target_names:
            continue
        linear = _as_linear(mod)
        if linear is None:
            continue
        # layers.*.self_attn.q_proj  or  model.layers.*.self_attn.q_proj
        parts = name.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break
        if layer_idx is None:
            continue
        n_layers = max(n_layers, layer_idx + 1)
        if leaf not in specs:
            specs[leaf] = ModuleSpec(
                leaf, int(linear.in_features), int(linear.out_features)
            )
    if not specs:
        raise ValueError(f"No target modules {target_names} found on base model")
    ordered = [specs[n] for n in target_names if n in specs]
    return n_layers, ordered

#hypermodulator class
class HyperModulator(nn.Module):
    """
    Generate LoRA A/B for every (layer, module) from task embedding.

    L: one head outputs both A and B (2 * r * d)
    M: shared head outputs A or B depending on AB embedding (r * d)
    S: head outputs one rank vector depending on AB + rank emb (d)
    """
    #dimensions taken from paper appendix F 
    def __init__(
        self,
        variant: str,
        n_layers: int,
        module_specs: List[ModuleSpec],
        rank: int = 8,
        task_in_dim: int = 1024,
        task_emb_dim: int = 64,
        module_emb_dim: int = 32,
        layer_emb_dim: int = 32,
        hidden_dim: int = 128,
        mlp_expand: int = 512,
        dropout: float = 0.05,
        use_text_task: bool = True,
        n_tasks: Optional[int] = None,
    ):
        super().__init__()
        variant = variant.upper()
        assert variant in {"L", "M", "S"}
        self.variant = variant
        self.n_layers = n_layers
        self.module_specs = module_specs
        self.module_names = [m.name for m in module_specs]
        self.rank = rank
        self.use_text_task = use_text_task
        self.hidden_dim = hidden_dim

        #task conditioning
        if use_text_task:
            self.task_encoder = nn.Sequential(
                nn.Linear(task_in_dim, task_emb_dim),
                nn.LayerNorm(task_emb_dim),
            )
        else:
            assert n_tasks is not None
            self.task_encoder = nn.Embedding(n_tasks, task_emb_dim)

        self.layer_emb = nn.Sequential(
            nn.Embedding(n_layers, layer_emb_dim),
            nn.LayerNorm(layer_emb_dim),
        )
        self.module_emb = nn.Sequential(
            nn.Embedding(len(module_specs), module_emb_dim),
            nn.LayerNorm(module_emb_dim),
        )

        mix_in = task_emb_dim + layer_emb_dim + module_emb_dim
        self.mixer = nn.Sequential(
            nn.Linear(mix_in, mlp_expand),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_expand, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.mlp1 = ResidualMLP(hidden_dim, mlp_expand, dropout)
        self.mlp2 = ResidualMLP(hidden_dim, mlp_expand, dropout)
        self.mlp3 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mlp_expand),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_expand, mlp_expand),
            nn.SiLU(),
        )
        self.out_dim = mlp_expand

        if variant in {"M", "S"}:
            #learnable AB embeddings per module: shape (hidden_dim,)
            self.ab_emb = nn.ParameterDict(
                {
                    name: nn.ParameterDict(
                        {
                            "A": nn.Parameter(torch.zeros(hidden_dim)),
                            "B": nn.Parameter(torch.zeros(hidden_dim)),
                        }
                    )
                    for name in self.module_names
                }
            )
        if variant == "S":
            self.rank_emb = nn.Sequential(
                nn.Embedding(rank, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

        #per-module output heads (d may differ for q vs v)
        self.heads = nn.ModuleDict()
        for spec in module_specs:
            #use max(in, out) as "d" for A (r x in) and B (out x r) packing
            d_pack = max(spec.in_features, spec.out_features)
            if variant == "L":
                out_size = 2 * rank * d_pack
            elif variant == "M":
                out_size = rank * d_pack
            else:
                out_size = d_pack
            self.heads[spec.name] = nn.Linear(mlp_expand, out_size)
            self._bias_hyper_init(self.heads[spec.name], variant, rank, d_pack)

        #store packing sizes for reshape
        self._pack_d = {s.name: max(s.in_features, s.out_features) for s in module_specs}
        self._specs = {s.name: s for s in module_specs}

    @staticmethod
    def _bias_hyper_init(head: nn.Linear, variant: str, rank: int, d: int) -> None:
        """Bias-HyperInit (Beck et al.): zero weights; A-like bias, B zero."""
        nn.init.zeros_(head.weight)
        with torch.no_grad():
            bias = head.bias
            bias.zero_()
            if variant == "L":
                # first half ~ A: U(-1/d, 1/d); second half B: 0
                half = bias.numel() // 2
                bound = 1.0 / d
                bias[:half].uniform_(-bound, bound)
            elif variant == "M":
                bound = 1.0 / (math.sqrt(2.0) * d)
                bias.uniform_(-bound, bound)
            else:  # S
                bound = 1.0 / (math.sqrt(2.0 * rank) * d)
                bias.uniform_(-bound, bound)

    def encode_task(self, task_input: torch.Tensor) -> torch.Tensor:
        """task_input: (B, task_in_dim) float OR (B,) long for one-hot path."""
        if self.use_text_task:
            return self.task_encoder(task_input)
        return self.task_encoder(task_input.long())

    def forward_slots(
        self,
        task_emb: torch.Tensor,
        layer_ids: torch.Tensor,
        module_ids: torch.Tensor,
        ab_ids: Optional[torch.Tensor] = None,
        rank_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward for a batch of slots.
        task_emb: (B, task_emb_dim)
        layer_ids / module_ids: (B,)
        returns: (B, out_features_of_head) — heads differ per module, so
        callers should group by module or use generate_lora().
        """
        raise RuntimeError("Use generate_lora() for full adapter generation")

    def _backbone(
        self,
        task_vec: torch.Tensor,
        layer_id: int,
        module_id: int,
        ab: Optional[str] = None,
        rank_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Single-slot backbone → mlp_expand features. task_vec: (task_emb_dim,)"""
        device = task_vec.device
        l = self.layer_emb(torch.tensor([layer_id], device=device)).squeeze(0)
        m = self.module_emb(torch.tensor([module_id], device=device)).squeeze(0)
        x = torch.cat([task_vec, l, m], dim=-1)
        h = self.mixer(x)
        h = self.mlp1(h)
        if self.variant in {"M", "S"} and ab is not None:
            name = self.module_names[module_id]
            h = h + self.ab_emb[name][ab]
        h = self.mlp2(h)
        if self.variant == "S" and rank_id is not None:
            h = h + self.rank_emb(torch.tensor([rank_id], device=device)).squeeze(0)
        h = self.mlp3(h)
        return h

    def _backbone_batched(
        self,
        task_vecs: torch.Tensor,
        layer_ids: torch.Tensor,
        module_ids: torch.Tensor,
        ab: Optional[str] = None,
        rank_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Batched backbone. task_vecs: (B, D)."""
        l = self.layer_emb(layer_ids)
        m = self.module_emb(module_ids)
        x = torch.cat([task_vecs, l, m], dim=-1)
        h = self.mixer(x)
        h = self.mlp1(h)
        if self.variant in {"M", "S"} and ab is not None:
            # add per-row AB emb (module-dependent)
            add = torch.zeros_like(h)
            for mid, name in enumerate(self.module_names):
                mask = module_ids == mid
                if mask.any():
                    add[mask] = self.ab_emb[name][ab]
            h = h + add
        h = self.mlp2(h)
        if self.variant == "S" and rank_ids is not None:
            h = h + self.rank_emb(rank_ids)
        return self.mlp3(h)

    def generate_lora(
        self, task_input: torch.Tensor
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate full LoRA dict for one or a batch of task embeddings.

        task_input: (task_in_dim,) or (1, task_in_dim) or task id long scalar/vector
        Returns nested dict:
          {module_name: {"A": (n_layers, r, in_f), "B": (n_layers, out_f, r)}}
        For a batch of tasks (B>1), tensors gain a leading batch dim.
        """
        if task_input.dim() == 1 and (
            self.use_text_task or task_input.dtype.is_floating_point
        ):
            task_input = task_input.unsqueeze(0)
        elif task_input.dim() == 0:
            task_input = task_input.unsqueeze(0)

        task_vecs = self.encode_task(task_input)  # (B, task_emb_dim)
        B = task_vecs.size(0)
        device = task_vecs.device
        result: Dict[str, Dict[str, torch.Tensor]] = {}

        for mid, spec in enumerate(self.module_specs):
            pack_d = self._pack_d[spec.name]
            head = self.heads[spec.name]
            layer_ids = torch.arange(self.n_layers, device=device).repeat(B)
            module_ids = torch.full((B * self.n_layers,), mid, device=device)
            # expand task vecs per layer
            tv = task_vecs.unsqueeze(1).expand(-1, self.n_layers, -1).reshape(
                B * self.n_layers, -1
            )

            if self.variant == "L":
                h = self._backbone_batched(tv, layer_ids, module_ids)
                out = head(h).view(B, self.n_layers, 2, self.rank, pack_d)
                A_pack = out[:, :, 0]  # (B, L, r, pack_d)
                B_pack = out[:, :, 1]
            elif self.variant == "M":
                hA = self._backbone_batched(tv, layer_ids, module_ids, ab="A")
                hB = self._backbone_batched(tv, layer_ids, module_ids, ab="B")
                A_pack = head(hA).view(B, self.n_layers, self.rank, pack_d)
                B_pack = head(hB).view(B, self.n_layers, self.rank, pack_d)
            else:  # S — one rank at a time
                A_ranks, B_ranks = [], []
                for ri in range(self.rank):
                    rids = torch.full((B * self.n_layers,), ri, device=device)
                    hA = self._backbone_batched(
                        tv, layer_ids, module_ids, ab="A", rank_ids=rids
                    )
                    hB = self._backbone_batched(
                        tv, layer_ids, module_ids, ab="B", rank_ids=rids
                    )
                    A_ranks.append(head(hA).view(B, self.n_layers, 1, pack_d))
                    B_ranks.append(head(hB).view(B, self.n_layers, 1, pack_d))
                A_pack = torch.cat(A_ranks, dim=2)
                B_pack = torch.cat(B_ranks, dim=2)

            # unpack to true A (r, in) and B (out, r)
            A = A_pack[..., : spec.in_features]  # (B, L, r, in)
            # B_pack stored as (B, L, r, pack_d); true B is (out, r)
            Bmat = B_pack[..., : spec.out_features].transpose(-1, -2)  # (B, L, out, r)

            if B == 1:
                result[spec.name] = {"A": A.squeeze(0), "B": Bmat.squeeze(0)}
            else:
                result[spec.name] = {"A": A, "B": Bmat}

        return result

    def flatten_lora(
        self, lora: Dict[str, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Concatenate all A/B into one vector (for recon loss)."""
        chunks = []
        for name in self.module_names:
            chunks.append(lora[name]["A"].reshape(-1))
            chunks.append(lora[name]["B"].reshape(-1))
        return torch.cat(chunks)

#text embedder for task conditioning 
class TaskEmbedder(nn.Module):
    """Frozen sentence encoder → CLS / pooled vector."""

    def __init__(self, model_name: str = "Alibaba-NLP/gte-large-en-v1.5"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.out_dim = self.encoder.config.hidden_size

    @torch.no_grad()
    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        out = self.encoder(**toks)
        # Mean pool (works for MiniLM); CLS also fine for gte-style models
        hidden = out.last_hidden_state
        mask = toks["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)


#inject loRA into base model
class DynamicLoRALinear(nn.Module):
    """Wraps nn.Linear with injectable low-rank A (r, in), B (out, r).

    A/B are plain attributes (not Parameters) so hypernetwork grads flow
    through the LoRA path during SFT while base weights stay frozen.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / math.sqrt(rank)  # RSLoRA-style
        self.in_features = base.in_features
        self.out_features = base.out_features
        self._A: Optional[torch.Tensor] = None
        self._B: Optional[torch.Tensor] = None
        self.enabled = True

    def set_lora(self, A: torch.Tensor, B: torch.Tensor) -> None:
        #keep autograd graph; only move device (dtype cast in forward)
        self._A = A.to(device=self.base.weight.device)
        self._B = B.to(device=self.base.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.base.weight, self.base.bias)
        if self.enabled and self._A is not None and self._B is not None:
            A = self._A.to(dtype=x.dtype)
            B = self._B.to(dtype=x.dtype)
            y = y + (x @ A.transpose(-1, -2) @ B.transpose(-1, -2)) * self.scaling
        return y


def attach_dynamic_lora(
    model: nn.Module,
    target_modules: List[str],
    rank: int,
    alpha: float = 16.0,
) -> Dict[Tuple[int, str], DynamicLoRALinear]:
    """Replace target Linears with DynamicLoRALinear; return lookup table."""
    lookup: Dict[Tuple[int, str], DynamicLoRALinear] = {}
    #find parent modules that own the Linear children
    to_replace = []
    for name, mod in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in target_modules or not isinstance(mod, nn.Linear):
            continue
        parts = name.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break
        if layer_idx is None:
            continue
        parent_name = name.rsplit(".", 1)[0]
        to_replace.append((parent_name, leaf, layer_idx, mod))

    for parent_name, leaf, layer_idx, linear in to_replace:
        parent = model.get_submodule(parent_name)
        wrapped = DynamicLoRALinear(linear, rank=rank, alpha=alpha)
        setattr(parent, leaf, wrapped)
        lookup[(layer_idx, leaf)] = wrapped
    return lookup


def inject_generated_lora(
    lookup: Dict[Tuple[int, str], DynamicLoRALinear],
    lora: Dict[str, Dict[str, torch.Tensor]],
) -> None:
    """Write generated A/B tensors into DynamicLoRALinear wrappers."""
    for name, mats in lora.items():
        A = mats["A"]  # (L, r, in) or (r, in) for single layer? always (L, r, in)
        B = mats["B"]  # (L, out, r)
        n_layers = A.size(0)
        for li in range(n_layers):
            key = (li, name)
            if key in lookup:
                lookup[key].set_lora(A[li], B[li])


def clear_lora(lookup: Dict[Tuple[int, str], DynamicLoRALinear]) -> None:
    for wrap in lookup.values():
        wrap._A = None
        wrap._B = None
