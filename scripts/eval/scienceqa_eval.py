"""
ScienceQA evaluation script.

Evaluates model accuracy on the ScienceQA closed-choice, text-only test set.
The model generates a chain-of-thought solution ending with
"The answer is (X)." and we extract the letter to compute accuracy.

Usage:
    python scripts/eval/scienceqa_eval.py \
        pretrained_model_name_or_path=<path_to_ckpt> \
        ...
"""

import json
import os
import re
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from datasets import load_dataset
from scripts.utils import (
    count_parameters,
    format_number,
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
    set_seed,
)
from src.utils import fsspec_exists

_ANSWER_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]
_ANSWER_PATTERN = re.compile(r"The answer is \(([A-H])\)")


def _format_question(question: str, choices: list[str], hint: str | None = None) -> str:
    parts = [question]
    if hint:
        parts.append(f"Hint: {hint}")
    for idx, choice in enumerate(choices):
        parts.append(f"({_ANSWER_LETTERS[idx]}) {choice}")
    return "\n".join(parts)


def _extract_answer(text: str) -> str | None:
    """Extract the predicted answer letter from generated text."""
    matches = _ANSWER_PATTERN.findall(text)
    if matches:
        return matches[-1]
    return None


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="kodcode_eval_config",
)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model_name_or_path = cfg.pretrained_model_name_or_path
    ckpt_file = cfg.get("ckpt_file", "best-rank0.pt")
    load_ema = cfg.get("load_ema_weights", True)

    if "fsdp" in pretrained_model_name_or_path:
        load_ema = False

    if fsspec_exists(os.path.join(pretrained_model_name_or_path, "config.yaml")):
        model = load_model_from_ckpt_dir_path(
            path_to_ckpt_dir=pretrained_model_name_or_path,
            load_ema_weights=load_ema,
            ckpt_file=ckpt_file,
            device=device,
        )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True
        )
    model = model.to(device)
    model.eval()
    print(f"Num. params: {format_number(count_parameters(model, trainable=False))}")

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    # --- Load ScienceQA test set ---
    test_size = cfg.get("scienceqa_test_size", 1000)
    sampling_seed = cfg.get("scienceqa_sampling_seed", 42)
    num_samples = cfg.get("scienceqa_num_samples", None)

    test_ds = load_dataset("derek-thomas/ScienceQA", split="test", trust_remote_code=True)
    test_ds = test_ds.filter(
        lambda x: x["image"] is None and x["task"] == "closed choice"
    )
    if len(test_ds) > test_size:
        rng = np.random.RandomState(sampling_seed)
        indices = sorted(
            rng.choice(len(test_ds), size=test_size, replace=False).tolist()
        )
        test_ds = test_ds.select(indices)

    if num_samples is not None:
        test_ds = test_ds.select(range(min(num_samples, len(test_ds))))

    print(f"Evaluating on {len(test_ds)} ScienceQA problems")

    # --- Generation config ---
    gen_kwargs = {}
    if hasattr(cfg, "gen_kwargs"):
        gen_kwargs = hydra.utils.instantiate(cfg.gen_kwargs)
        if isinstance(gen_kwargs, DictConfig):
            from omegaconf import OmegaConf

            gen_kwargs = OmegaConf.to_container(gen_kwargs, resolve=True)


    output_path = cfg.get("output_path", "./scienceqa_output")
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    source_prompt_text = (
        "The following is a multiple choice question. Think step by step "
        "and then give your final answer.\n\n"
    )
    target_prompt_text = "Answer: "

    results_per_problem = []
    tputs = []
    total_correct = 0
    total_problems = 0
    total_generated_tokens = 0
    total_accepted_tokens = 0
    total_accepted_lengths = []
    total_accept_counts = 0
    warmup = 5

    # Per-subject accuracy tracking
    subject_stats: dict[str, dict[str, int]] = {}

    for idx in tqdm(range(len(test_ds)), desc="Evaluating ScienceQA"):
        example = test_ds[idx]
        question = example["question"]
        choices = example["choices"]
        hint = example.get("hint", "")
        gold_answer_idx = example["answer"]
        gold_letter = _ANSWER_LETTERS[gold_answer_idx]
        subject = example.get("subject", "unknown")

        q_text = _format_question(question, choices, hint)

        is_seed = "SEED" in type(model).__name__
        # SEED uses bidirectional attention to encode the prompt during training.
        if is_seed:
            ctx = (
                (tokenizer.bos_token or "")
                + source_prompt_text
                + q_text
                + (tokenizer.eos_token or "")
            )
        else:
            ctx = (
                (tokenizer.bos_token or "")
                + source_prompt_text
                + q_text
                + (tokenizer.eos_token or "")
                + target_prompt_text
            )

        prefix_tokens = tokenizer(ctx, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        is_layerskip_speculative = (
            "LayerSkip" in type(model).__name__
            and gen_kwargs.get("assistant_early_exit") is not None
        )

        if is_seed or is_layerskip_speculative:
            sample, (generated_tokens, accepted_tokens), (
                accepted_lengths,
                accept_counts,
            ), (draft_time_s, all_time_s) = model.generate(
                inputs=prefix_tokens,
                disable_pbar=False,
                **gen_kwargs,
            )
        else:
            sample = model.generate(
                inputs=prefix_tokens,
                disable_pbar=False,
                **gen_kwargs,
            )

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_s = start_event.elapsed_time(end_event) / 1000
        num_new_tokens = sample.numel() - prefix_tokens.numel()

        if not (is_seed or is_layerskip_speculative):
            generated_tokens = max(int(num_new_tokens), 1)
            accepted_tokens = generated_tokens
            accepted_lengths = [1]
            accept_counts = 1

        total_generated_tokens += generated_tokens
        total_accepted_tokens += accepted_tokens
        total_accepted_lengths.extend(accepted_lengths)
        total_accept_counts += accept_counts

        tput = num_new_tokens / elapsed_time_s if elapsed_time_s > 0 else 0
        if idx >= warmup:
            tputs.append(tput)

        generated_text = tokenizer.decode(
            sample[0, prefix_tokens.shape[1]:], skip_special_tokens=True
        )

        # Strip at common stop sequences
        for stop_seq in [tokenizer.eos_token, "<|eot_id|>"]:
            if stop_seq:
                generated_text = generated_text.split(stop_seq)[0]

        predicted_letter = _extract_answer(generated_text)
        is_correct = predicted_letter == gold_letter

        if is_correct:
            total_correct += 1
        total_problems += 1

        # Per-subject tracking
        if subject not in subject_stats:
            subject_stats[subject] = {"correct": 0, "total": 0}
        subject_stats[subject]["total"] += 1
        if is_correct:
            subject_stats[subject]["correct"] += 1

        results_per_problem.append(
            {
                "idx": idx,
                "question": question,
                "choices": choices,
                "hint": hint,
                "gold_answer": gold_letter,
                "predicted_answer": predicted_letter,
                "is_correct": is_correct,
                "generated_text": generated_text,
                "subject": subject,
            }
        )

        accuracy = total_correct / total_problems
        if idx >= warmup and tputs:
            print(
                f"\n[{idx + 1}/{len(test_ds)}] "
                f"Accuracy: {total_correct}/{total_problems} = {accuracy:.2%} | "
                f"Thput: {np.mean(tputs):.2f} +/- {np.std(tputs):.2f} tok/s"
            )
        else:
            print(
                f"\n[{idx + 1}/{len(test_ds)}] "
                f"Accuracy: {total_correct}/{total_problems} = {accuracy:.2%} | "
                f"Thput: {tput:.2f} tok/s"
            )
        if total_generated_tokens > 0:
            print(
                f"Total generated tokens: {total_generated_tokens}, "
                f"Total accepted tokens: {total_accepted_tokens}, "
                f"Acceptance rate: "
                f"{total_accepted_tokens / total_generated_tokens:.2%}"
            )
        if total_accept_counts > 0:
            print(
                f"Average accepted length: "
                f"{np.sum(total_accepted_lengths) / total_accept_counts:.2f}"
            )
        print(f"  Gold: ({gold_letter})  Predicted: ({predicted_letter})")
        print(f"  Question: {ctx}")
        print(f"  Generated: {generated_text}")

    # --- Aggregate metrics ---
    final_accuracy = total_correct / total_problems if total_problems > 0 else 0.0

    metrics: dict[str, Any] = {
        "overall_accuracy": final_accuracy,
        "total_correct": total_correct,
        "total_problems": total_problems,
        "per_subject": {
            s: {
                "accuracy": (
                    v["correct"] / v["total"] if v["total"] > 0 else 0.0
                ),
                "correct": v["correct"],
                "total": v["total"],
            }
            for s, v in subject_stats.items()
        },
    }
    if tputs:
        metrics["throughput_mean_tok_s"] = float(np.mean(tputs))
        metrics["throughput_std_tok_s"] = float(np.std(tputs))
    if total_generated_tokens > 0:
        metrics["acceptance_rate"] = total_accepted_tokens / total_generated_tokens
    if total_accept_counts > 0:
        metrics["average_accepted_length"] = float(
            np.sum(total_accepted_lengths) / total_accept_counts
        )

    with open(os.path.join(output_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(output_path, "results.json"), "w") as f:
        json.dump(results_per_problem, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SCIENCEQA EVALUATION RESULTS")
    print("=" * 60)
    print(
        f"Overall Accuracy: {final_accuracy:.2%} "
        f"({total_correct}/{total_problems})"
    )
    for s, v in sorted(subject_stats.items()):
        s_acc = v["correct"] / v["total"] if v["total"] > 0 else 0.0
        print(f"  {s}: {s_acc:.2%} ({v['correct']}/{v['total']})")
    if tputs:
        print(
            f"Throughput: {np.mean(tputs):.2f} +/- {np.std(tputs):.2f} tok/s"
        )
    if total_generated_tokens > 0:
        print(
            f"Acceptance rate: "
            f"{total_accepted_tokens / total_generated_tokens:.2%}"
        )
    if total_accept_counts > 0:
        print(
            f"Average accepted length: "
            f"{np.sum(total_accepted_lengths) / total_accept_counts:.2f}"
        )
    print(f"Results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    register_useful_resolvers()
    main()
