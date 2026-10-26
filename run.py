#!/usr/bin/env python3
""" Text-to-LoRA recreation.

Usage:
  python run.py descriptions
  python run.py oracles
  python run.py recon
  python run.py sft
  python run.py multitask
  python run.py eval
  python run.py all         
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from t2l.data import (
    DEFAULT_DESCRIPTIONS,
    build_description_bank,
    load_config,
    prepare_sni_split,
)
from t2l.eval import run_eval, viz_tsne
from t2l.train import train_multitask_lora, train_oracles, train_recon, train_sft


def cmd_descriptions(cfg):
    benches = list(
        dict.fromkeys(cfg["oracle"]["tasks"] + cfg["eval"]["benches"])
    )
    build_description_bank(
        benches, cfg["recon"]["descriptions_per_task"], Path("data/bench_descriptions.json")
    )
    # SNI bank after / with prepare
    manifest = prepare_sni_split(
        cfg["sft"]["n_train_tasks"],
        cfg["sft"]["n_holdout_tasks"],
        cfg["sft"]["samples_per_task"],
        Path("data/sni"),
    )
    build_description_bank(
        manifest["train_tasks"] + manifest["holdout_tasks"],
        cfg["sft"]["descriptions_per_task"],
        Path("data/sni_descriptions.json"),
        sni_definitions=manifest.get("definitions"),
    )
    print("Description banks written under data/")


def main():
    p = argparse.ArgumentParser(description="Text-to-LoRA class project")
    p.add_argument(
        "command",
        choices=[
            "descriptions",
            "oracles",
            "recon",
            "sft",
            "multitask",
            "eval",
            "viz",
            "all",
        ],
    )
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    cfg = load_config(ROOT / args.config)

    if args.command == "descriptions":
        cmd_descriptions(cfg)
    elif args.command == "oracles":
        train_oracles(cfg)
    elif args.command == "recon":
        train_recon(cfg)
    elif args.command == "sft":
        train_sft(cfg)
    elif args.command == "multitask":
        train_multitask_lora(cfg)
    elif args.command == "eval":
        run_eval(cfg)
    elif args.command == "viz":
        viz_tsne(cfg)
    elif args.command == "all":
        cmd_descriptions(cfg)
        train_oracles(cfg)
        train_recon(cfg)
        train_sft(cfg)
        train_multitask_lora(cfg)
        run_eval(cfg)


if __name__ == "__main__":
    main()
