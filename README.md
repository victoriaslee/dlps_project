# Text-to-LoRA class project

This project is a smaller implementation of the Text-to-LoRA idea from
[the original paper](https://arxiv.org/abs/2506.06105). The main experiment
uses a hypernetwork to generate LoRA adapters from short descriptions of a
task. 

The project is intentionally scaled down so it can run on an Apple Silicon
machine or a consumer GPU. The default configuration uses Qwen2.5-1.5B,
MiniLM task embeddings, rank-8 LoRA adapters, and the `q_proj` and `v_proj`
attention modules.

## What is included

There are two ways to train the hypernetwork:

1. **Reconstruction:** train ordinary LoRA adapters for several benchmark
   tasks, then train the hypernetwork to reproduce those adapters.
2. **SFT:** keep the base language model frozen and train the hypernetwork
   through the language-model loss while it generates the LoRA adapters.

The code also includes a regular multi-task LoRA baseline, benchmark
evaluation, a description ablation, and a t-SNE plot of task-encoder outputs.
The hypernetwork supports the L, M, and S output variants described in the
paper; the default run uses M as does the paper.

## Project layout

```text
project/
  config.yaml       # model and training settings
  report.pdf        # report
  run.py            # command-line entrypoint
  requirements.txt  # Python dependencies
  t2l/
    model.py        # hypernetwork, text encoder, and dynamic LoRA layers
    data.py         # prompts, datasets, descriptions, and LoRA I/O
    train.py        # oracle, reconstruction, SFT, and baseline training
    eval.py         # evaluation, ablation, and t-SNE
  data/             # cached descriptions and task examples
  artifacts/        # checkpoints, adapters, and evaluation results
```

## Installation

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The first run downloads the base model, MiniLM, and any requested datasets
from Hugging Face. Internet access and sufficient disk space are required for
those downloads.

The code uses CUDA when available, then Apple MPS, and otherwise falls back to
CPU. CPU training will be very slow.

## Running the project

Each stage should be run sequentially as follows:

```bash
# Create benchmark and SFT task descriptions
python run.py descriptions

# Train one ordinary LoRA adapter for each oracle benchmark
python run.py oracles

# Train the hypernetwork to reconstruct the oracle adapters
python run.py recon

# Train the hypernetwork end-to-end through the language model
python run.py sft

# Optional: train one LoRA on the combined oracle-task data
python run.py multitask

# Evaluate the available models and create the t-SNE plot
python run.py eval
```

To run all stages in sequence:

```bash
python run.py all
```

The full run takes ~2 hours. For a shorter test run, reduce
`oracle.max_steps`, `recon.max_steps`, `sft.max_steps`, or
`eval.max_samples` in [`config.yaml`](config.yaml).

## SFT data options

By default, the SFT stage creates a small synthetic multi-task corpus. With
the current configuration this gives 12 training tasks and 4 holdout tasks.
This is useful for checking that the pipeline runs, but it is not a substitute
for the larger SNI experiment in the paper.

To use the small set of real Hugging Face classification datasets instead,
run:

```bash
T2L_USE_HF_TASKS=1 python run.py descriptions
```

To try the Lots-of-LoRAs/SNI task names, run:

```bash
T2L_DOWNLOAD_SNI=1 python run.py descriptions
```

The SNI option is slower and depends on the corresponding datasets being
available. Existing files in `data/sni/` are reused when a manifest is
already present.

## Current configuration

| Setting | Current default |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Task embedder | `sentence-transformers/all-MiniLM-L6-v2` |
| Hypernetwork variant | M |
| LoRA rank | 8 |
| LoRA target modules | `q_proj`, `v_proj` |
| Oracle tasks | BoolQ, ARC-Easy, PIQA, OpenBookQA |
| SFT split | 12 training tasks, 4 holdout tasks |
| Evaluation examples | 500 per benchmark |

The text encoder is frozen and mean-pools its token representations. The
base language model is also frozen when the hypernetwork is trained for SFT and
only the hypernetwork is updated.

## Results currently in `artifacts/`

The saved evaluation results are in
[`artifacts/eval_results.json`](artifacts/eval_results.json). The reconstruction
run is close to the ordinary oracle LoRA performance on the four benchmarks.
The initial SFT run performs much worse, largely because the small synthetic
training data does not match the Boolean and multiple-choice output formats
used at evaluation time. The current configuration has been revised, but the
revised SFT run is not included in the saved results yet.

The t-SNE figure is saved as
[`artifacts/tsne_task_encoder.png`](artifacts/tsne_task_encoder.png).

## Citation

```text
Charakorn et al., Text-to-LoRA: Instant Transformer Adaption, 2025.
https://arxiv.org/abs/2506.06105
```
