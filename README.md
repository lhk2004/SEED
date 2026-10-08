# Self-Speculative Decoding via Implicit Encoder–Decoder

Official implementation of [**SEED: Self-Speculative Decoding via Implicit Encoder–Decoder**](https://arxiv.org/abs/2609.36590).

SEED reinterprets a standard decoder-only transformer as an implicit encoder–decoder:
the first layers (encoder) build deep contextual representations, and the last few
layers (decoder) emit tokens from them. Verification runs through the full model and
caches the contextual representations of the verified prefix. Between verifications, the
lightweight decoder drafts several tokens autoregressively, conditioned on those cached
representations. This makes drafting cheap without sacrificing draft quality.

## Overview

The repo trains and evaluates four kinds of models on four task datasets.

| Model | Description | Hydra config |
| --- | --- | --- |
| AR | Standard autoregressive fine-tuning | [`model=ar`](configs/model/ar.yaml) |
| SEED (ours) | Self-speculative encoder–decoder | [`model=seed`](configs/model/seed.yaml) |
| E2D2 | Encoder–decoder block diffusion ([Arriola et al.](https://arxiv.org/abs/2510.22852)) | [`model=e2d2`](configs/model/e2d2.yaml) |
| LayerSkip | Early-exit self-speculative decoding ([Elhoushi et al.](https://arxiv.org/abs/2404.16710)) | [`model=layerskip`](configs/model/layerskip.yaml) |

| Task | Domain | Hugging Face dataset | Metric script |
| --- | --- | --- | --- |
| GSM8K | Math reasoning | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) | [`harness_eval.py`](scripts/eval/harness_eval.py) (lm-eval-harness) |
| KodCode | Coding | [`KodCode/KodCode-V1-SFT-R1`](https://huggingface.co/datasets/KodCode/KodCode-V1-SFT-R1) | [`kodcode_eval.py`](scripts/eval/kodcode_eval.py) |
| ScienceQA | General knowledge | [`derek-thomas/ScienceQA`](https://huggingface.co/datasets/derek-thomas/ScienceQA) | [`scienceqa_eval.py`](scripts/eval/scienceqa_eval.py) |
| CNN/DailyMail | Summarization | [`abisee/cnn_dailymail`](https://huggingface.co/datasets/abisee/cnn_dailymail) | [`seq2seq_eval.py`](scripts/eval/seq2seq_eval.py) |

All models are fine-tuned from Qwen3 base models (`Qwen/Qwen3-1.7B-Base` by default, but `Qwen/Qwen3-4B-Base` is supported as well).

### Baselines not included in this repo

Our paper also compares against EAGLE-3, Apple MTP, SWIFT, and DEL. This repo does not
include them, because we ran each one with an existing codebase instead of
reimplementing it:

- **EAGLE-3**: the official [EAGLE repository](https://github.com/SafeAILab/EAGLE), with
  the public AngelSlim drafter checkpoints
  ([Qwen3-1.7B](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3),
  [Qwen3-4B](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3)).
- **Apple MTP**: there is no official code release, so we used the unofficial
  [MTP-GLoRA](https://github.com/siihwanpark/MTP-GLoRA) implementation, modified to match
  the paper's algorithm more closely.
- **SWIFT** and **DEL**: the authors' released implementations. SWIFT runs on our
  fine-tuned AR checkpoints and DEL on our fine-tuned LayerSkip checkpoints, both
  trained with this repo.

## Setup

### Environment

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate seed-env
```

All shell scripts in [`bash_scripts`](bash_scripts) call [`setup_env.sh`](setup_env.sh).
It activates the environment, sets `PYTHONPATH` to the repo root, and puts the Hugging
Face cache under `./.hf_cache`.

### W&B and Hugging Face tokens

`setup_env.sh` also sources `${HOME}/setup_seed.sh` for credentials. Create that file
with your own tokens:

```bash
# W&B setup
export WANDB__SERVICE_WAIT=600
export WANDB_ENTITY="<WANDB_ENTITY>"
export WANDB_API_KEY="<WANDB_API_KEY>"

# Hugging Face setup
export HUGGINGFACE_TOKEN="<HF_TOKEN>"
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

You can find your W&B key [here](https://wandb.ai/authorize) and create a Hugging Face
token [here](https://huggingface.co/settings/tokens). Runs are logged to the W&B project
`seed` ([`configs/composer/loggers/wandb.yaml`](configs/composer/loggers/wandb.yaml)).

## Training

There is one training script per model and task, named
`bash_scripts/run_train_<model>_<task>.sh`:

| | GSM8K | KodCode | ScienceQA | CNN/DailyMail |
| --- | --- | --- | --- | --- |
| AR | `run_train_ar_gsm8k.sh` | `run_train_ar_kodcode.sh` | `run_train_ar_scienceqa.sh` | `run_train_ar_cnn.sh` |
| SEED | `run_train_seed_gsm8k.sh` | `run_train_seed_kodcode.sh` | `run_train_seed_scienceqa.sh` | `run_train_seed_cnn.sh` |
| E2D2 | `run_train_e2d2_gsm8k.sh` | `run_train_e2d2_kodcode.sh` | `run_train_e2d2_scienceqa.sh` | `run_train_e2d2_cnn.sh` |
| LayerSkip | `run_train_layerskip_gsm8k.sh` | `run_train_layerskip_kodcode.sh` | `run_train_layerskip_scienceqa.sh` | `run_train_layerskip_cnn.sh` |

The scripts must be launched from inside `bash_scripts/`, and they use every GPU listed
in `CUDA_VISIBLE_DEVICES`:

```bash
cd bash_scripts
CUDA_VISIBLE_DEVICES=0 bash run_train_seed_gsm8k.sh
```

Each run writes to `outputs/<RUN_NAME>/` at the repo root, and the best checkpoint is
saved as `outputs/<RUN_NAME>/checkpoints/best-rank0.pt`. To save elsewhere, change
`hydra.run.dir` in the script.

> [!NOTE]
> **The "encoder" means something different in the code than in the paper.** This code
> follows E2D2's terminology, where the *encoder* is the full pretrained model and
> the *decoder* is its top `N_DECODER_LAYERS` layers reused for drafting. However, in
> the paper, the *encoder* is only the layers below the decoder. For example, with
> `Qwen3-1.7B-Base`, the scripts set `N_ENCODER_LAYERS=28` and `N_DECODER_LAYERS=2`:
> in the paper's terms, that is a 26-layer encoder followed by a 2-layer decoder, not
> 28 + 2 layers.

The main SEED settings at the top of each SEED script are:

- `N_ENCODER_LAYERS` / `N_DECODER_LAYERS`: the number of layers in the full model
  (the code's "encoder"), and how many of its top layers form the drafter. Our
  fine-tuned `Qwen3-1.7B-Base` uses `28` / `2`, so the last 2 of its 28 layers act as
  the drafter.
- `BLOCK_SIZE`: draft length used during training.
- `DECODER_LOSS_LAMBDA`: weight of the drafting loss relative to the next-token loss.
- `TIE_WEIGHTS`: share parameters between the decoder and the top layers of the model.

Training is driven by [Composer](https://github.com/mosaicml/composer) and configured
with [Hydra](https://hydra.cc). Any field in [`configs/config.yaml`](configs/config.yaml)
can be overridden on the command line.

## Evaluation

Each task has its own evaluation script:

| Task | Script |
| --- | --- |
| GSM8K | [`run_lm_eval_harness.sh`](bash_scripts/run_lm_eval_harness.sh) |
| KodCode | [`evaluate_kodcode.sh`](bash_scripts/evaluate_kodcode.sh) |
| ScienceQA | [`evaluate_scienceqa.sh`](bash_scripts/evaluate_scienceqa.sh) |
| CNN/DailyMail | [`run_seq2seq_eval_cnndm.sh`](bash_scripts/run_seq2seq_eval_cnndm.sh) |

Every script has one section per model type (AR, SEED, E2D2, LayerSkip). To evaluate a
checkpoint:

1. Uncomment the section for your model type, and comment out the others.
2. Set `MODEL_PATH` to the training run directory, e.g.
   `MODEL_PATH="${PWD}/outputs/<RUN_NAME>"` (the folder that contains `checkpoints/`).
   The script `cd`s to the repo root before reading it, so `${PWD}` is the repo root.
3. Set `QWEN_MODEL` to the tokenizer of the base model you fine-tuned.
4. Run the script from inside `bash_scripts/`:

   ```bash
   cd bash_scripts
   CUDA_VISIBLE_DEVICES=0 bash run_lm_eval_harness.sh
   ```

The scripts report task accuracy (or ROUGE for CNN/DailyMail), decoding throughput in tokens/s, and, for speculative methods, the draft quality statistics. Results are written under `MODEL_PATH`, next to the checkpoints.

## Acknowledgements

This repository is built on the [E2D2 codebase](https://github.com/kuleshov-group/e2d2). We thank the authors for releasing their code.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{lin2026seed,
  title     = {{SEED}: Self-Speculative Decoding via Implicit Encoder-Decoder},
  author    = {Lin, Hankun and Pynadath, Patrick and Zhang, Ruqi},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://arxiv.org/abs/2609.36590},
}
```
