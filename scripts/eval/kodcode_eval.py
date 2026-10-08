"""
KodCode evaluation script.

This script evaluates model-generated code on the KodCode benchmark by executing
the generated solutions against test cases (pytest-style or stdio-style).

Uses the KodCode/KodCode-V1-SFT-R1 dataset. We filter for version=="v1.1"
samples, randomly sample half (with a fixed seed), then use the last 1k of
those as the test set.

Usage:
    python scripts/eval/kodcode_eval.py \\
        pretrained_model_name_or_path=<path_to_ckpt> \\
        ...

The script reuses the same model loading and generation infrastructure as the
other evaluation scripts, and scores completions by running KodCode's unit tests.
"""

import ast
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
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


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException("Timed out!")


def _run_kodcode_tests(
    generated_code: str,
    test_code: str,
    timeout: int = 10,
) -> dict[str, Any]:
    """Run generated code against KodCode pytest-style test cases.

    KodCode tests use `from solution import <func_name>`, so we:
    1. Write the generated code to a temporary 'solution.py' file
    2. Execute each test function in the test code

    Args:
        generated_code: The generated Python code string.
        test_code: The pytest-style test code from the dataset.
        timeout: Maximum seconds for running all tests.

    Returns:
        Dict with 'passed', 'total', 'all_passed', and 'details'.
    """
    results = {"passed": 0, "total": 0, "all_passed": False, "details": []}

    if not generated_code.strip() or not test_code.strip():
        return results

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write the generated code as solution.py
        solution_path = os.path.join(tmpdir, "solution.py")
        with open(solution_path, "w") as f:
            f.write(generated_code)

        # Add tmpdir to sys.path so `from solution import ...` works
        sys.path.insert(0, tmpdir)

        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)

            # Parse test functions from the test code
            test_functions = _extract_test_functions(test_code)

            if not test_functions:
                # If we can't parse individual test functions, try running the
                # whole test module
                results["total"] = 1
                try:
                    namespace = {}
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        exec(test_code, namespace)
                    results["passed"] = 1
                    results["details"].append({"name": "all_tests", "passed": True})
                except Exception as e:
                    results["details"].append(
                        {"name": "all_tests", "passed": False, "error": str(e)}
                    )
            else:
                # Build a combined module: imports + test functions run sequentially
                # First, extract the import lines
                import_lines = []
                other_lines = []
                for line in test_code.split("\n"):
                    stripped = line.strip()
                    if stripped.startswith("import ") or stripped.startswith("from "):
                        import_lines.append(line)
                    else:
                        other_lines.append(line)

                import_block = "\n".join(import_lines)

                for test_name, test_body in test_functions:
                    results["total"] += 1
                    try:
                        # Build executable code: imports + test function def + call
                        exec_code = (
                            import_block
                            + "\n\n"
                            + test_body
                            + "\n\n"
                            + test_name
                            + "()\n"
                        )
                        namespace = {}
                        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                            exec(exec_code, namespace)
                        results["passed"] += 1
                        results["details"].append(
                            {"name": test_name, "passed": True}
                        )
                    except TimeoutException:
                        results["details"].append(
                            {"name": test_name, "passed": False, "error": "Timeout"}
                        )
                        # On timeout, mark remaining as failed
                        remaining_count = len(test_functions) - results["total"]
                        results["total"] += remaining_count
                        for _ in range(remaining_count):
                            results["details"].append(
                                {"name": "skipped", "passed": False, "error": "Timeout"}
                            )
                        break
                    except Exception as e:
                        results["details"].append(
                            {"name": test_name, "passed": False, "error": str(e)}
                        )

        except TimeoutException:
            if results["total"] == 0:
                results["total"] = 1
            results["details"].append(
                {"name": "timeout", "passed": False, "error": "Global timeout"}
            )
        except Exception as e:
            if results["total"] == 0:
                results["total"] = 1
            results["details"].append(
                {"name": "error", "passed": False, "error": str(e)}
            )
        finally:
            signal.alarm(0)
            # Clean up sys.path
            if tmpdir in sys.path:
                sys.path.remove(tmpdir)
            # Clean up cached solution module
            if "solution" in sys.modules:
                del sys.modules["solution"]

    results["all_passed"] = (
        results["total"] > 0 and results["passed"] == results["total"]
    )
    return results


def _extract_test_functions(test_code: str) -> list[tuple[str, str]]:
    """Extract individual test functions from pytest-style test code.

    Returns:
        List of (function_name, function_body_including_def) tuples.
    """
    import re

    functions = []
    lines = test_code.split("\n")
    current_func_name = None
    current_func_lines = []

    for line in lines:
        match = re.match(r"^def (test_\w+)\s*\(", line)
        if match:
            # Save previous function if any
            if current_func_name is not None:
                functions.append(
                    (current_func_name, "\n".join(current_func_lines))
                )
            current_func_name = match.group(1)
            current_func_lines = [line]
        elif current_func_name is not None:
            # Check if this line is part of the current function
            # (indented or empty line)
            if line.startswith((" ", "\t")) or line.strip() == "":
                current_func_lines.append(line)
            else:
                # Non-indented, non-empty line outside a function
                # End the current function
                if not line.strip().startswith(("import ", "from ", "#", "@")):
                    functions.append(
                        (current_func_name, "\n".join(current_func_lines))
                    )
                    current_func_name = None
                    current_func_lines = []
                else:
                    # It's an import/comment/decorator between functions
                    if current_func_name is not None:
                        functions.append(
                            (current_func_name, "\n".join(current_func_lines))
                        )
                        current_func_name = None
                        current_func_lines = []

    # Don't forget the last function
    if current_func_name is not None:
        functions.append((current_func_name, "\n".join(current_func_lines)))

    return functions


def _run_kodcode_stdio_tests(
    generated_code: str,
    test_str: str,
    timeout: int = 10,
) -> dict[str, Any]:
    """Run generated code against KodCode stdio-style test cases.

    For Online Judge style problems, the ``test`` field is a string
    representation of a list/dict of input/output pairs.
    Each pair has ``input`` and ``output`` (or ``expected_output``) keys.

    Args:
        generated_code: The generated Python code string.
        test_str: The stdio test specification string from the dataset.
        timeout: Maximum seconds per test case.

    Returns:
        Dict with 'passed', 'total', 'all_passed', and 'details'.
    """
    # print("-"*30)
    # print("Generated code:", generated_code)
    # print("Test string:", test_str)
    # print("-"*30)
    results = {"passed": 0, "total": 0, "all_passed": False, "details": []}

    if not generated_code.strip() or not test_str.strip():
        return results

    try:
        test_cases = ast.literal_eval(test_str)
    except Exception as e:
        results["total"] = 1
        results["details"].append(
            {"name": "stdio_parse", "passed": False, "error": f"Invalid test format: {e}"}
        )
        return results

    normalized_cases = []
    # Common KodCode Online Judge format:
    # {'stdin': [...], 'stdout': [...]} or {'input': [...], 'output': [...]}.
    stdin_list = test_cases.get("stdin", test_cases.get("input"))
    stdout_list = test_cases.get("stdout", test_cases.get("output", test_cases.get("expected_output")))
    if len(stdin_list) != len(stdout_list):
        results["total"] = 1
        results["details"].append(
            {
                "name": "stdio_parse",
                "passed": False,
                "error": "Mismatched stdin/stdout lengths",
            }
        )
        return results
    normalized_cases = [
        {"input": stdin_value, "output": stdout_value}
        for stdin_value, stdout_value in zip(stdin_list, stdout_list)
    ]
        

    if len(normalized_cases) == 0:
        results["total"] = 1
        results["details"].append(
            {
                "name": "stdio_parse",
                "passed": False,
                "error": "No valid stdio test cases found",
            }
        )
        return results

    for i, test_case in enumerate(normalized_cases):
        results["total"] += 1
        test_input = str(test_case.get("input", ""))
        expected_output = str(
            test_case.get("output", test_case.get("expected_output", ""))
        ).strip()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                solution_path = os.path.join(tmpdir, "solution.py")
                with open(solution_path, "w") as f:
                    f.write(generated_code)

                proc = subprocess.run(
                    [sys.executable, solution_path],
                    input=test_input,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                actual_output = proc.stdout.strip()
                # print("Test input:", test_input)
                # print("Actual output:", actual_output)
                # print("Expected output:", expected_output)
                # print("-"*30)
                if proc.returncode != 0:
                    results["details"].append(
                        {
                            "name": f"stdio_test_{i}",
                            "passed": False,
                            "error": f"Non-zero exit code {proc.returncode}: {proc.stderr[:200]}",
                        }
                    )
                elif actual_output == expected_output:
                    results["passed"] += 1
                    results["details"].append(
                        {"name": f"stdio_test_{i}", "passed": True}
                    )
                else:
                    results["details"].append(
                        {
                            "name": f"stdio_test_{i}",
                            "passed": False,
                            "error": (
                                f"Expected: {expected_output[:100]}, "
                                f"Got: {actual_output[:100]}"
                            ),
                        }
                    )
        except subprocess.TimeoutExpired:
            results["details"].append(
                {"name": f"stdio_test_{i}", "passed": False, "error": "Timeout"}
            )
        except Exception as e:
            results["details"].append(
                {"name": f"stdio_test_{i}", "passed": False, "error": str(e)}
            )

    results["all_passed"] = (
        results["total"] > 0 and results["passed"] == results["total"]
    )
    return results


def _normalize_generated_code(generated_text: str) -> str:
    """Normalize model output into executable Python code.

    Handles common output wrappers:
    - Markdown fenced code blocks
    - Leading explanatory prose before first code line
    """
    text = generated_text.strip()
    if not text:
        return text

    fenced_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if fenced_blocks:
        text = max(fenced_blocks, key=len).strip()

    lines = text.splitlines()
    code_prefixes = ("def ", "class ", "import ", "from ", "@")
    first_code_idx = None
    for idx, line in enumerate(lines):
        if line.lstrip().startswith(code_prefixes):
            first_code_idx = idx
            break

    if first_code_idx is not None:
        text = "\n".join(lines[first_code_idx:]).strip()

    text = re.sub(r"^```(?:python)?\s*", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="kodcode_eval_config",
)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    # --- Load model ---
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

    # --- Load tokenizer ---
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    # --- Load KodCode test set ---
    test_size = cfg.get("kodcode_test_size", 1000)
    sampling_seed = cfg.get("kodcode_sampling_seed", 42)
    num_samples = cfg.get("kodcode_num_samples", None)
    difficulty = cfg.get("kodcode_difficulty", None)

    full_dataset = load_dataset(
        "KodCode/KodCode-V1-SFT-R1",
        split="train",
        trust_remote_code=True,
    )
    # Filter to v1.1 samples only
    full_dataset = full_dataset.filter(
        lambda x: x.get("version", "") == "v1.1"
    )
    # Randomly sample half with a fixed seed for reproducibility
    rng = np.random.RandomState(sampling_seed)
    n_sampled = len(full_dataset) // 2
    sampled_indices = sorted(
        rng.choice(len(full_dataset), size=n_sampled, replace=False).tolist()
    )
    full_dataset = full_dataset.select(sampled_indices)
    # Take last test_size samples as test set
    train_size = len(full_dataset) - test_size
    kodcode_test = full_dataset.select(range(train_size, len(full_dataset)))

    if difficulty is not None:
        kodcode_test = kodcode_test.filter(
            lambda x: x.get("gpt_difficulty", "") == difficulty
        )
    if num_samples is not None:
        kodcode_test = kodcode_test.select(
            range(min(num_samples, len(kodcode_test)))
        )

    print(
        f"Evaluating on {len(kodcode_test)} KodCode problems "
        f"(difficulty={difficulty})"
    )

    # --- Generation config ---
    gen_kwargs = {}
    if hasattr(cfg, "gen_kwargs"):
        gen_kwargs = hydra.utils.instantiate(cfg.gen_kwargs)
        if isinstance(gen_kwargs, DictConfig):
            from omegaconf import OmegaConf

            gen_kwargs = OmegaConf.to_container(gen_kwargs, resolve=True)


    # --- Output dir ---
    output_path = cfg.get("output_path", "./kodcode_output")
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # --- Generate and evaluate ---
    results_per_problem = []
    tputs = []
    total_correct = 0
    total_correct_raw = 0
    recovered_by_normalization = 0
    total_problems = 0
    total_generated_tokens = 0
    total_accepted_tokens = 0
    total_accepted_lengths = []
    total_accept_counts = 0
    warmup = 5

    for idx in tqdm(range(len(kodcode_test)), desc="Evaluating KodCode"):
        example = kodcode_test[idx]
        question = example["question"]
        test_code = example.get("test", "")
        style = example.get("style", "Instruct")
        ref_solution = example.get("solution", "")

        # Same prompt format as used in training.
        prompt_text = (
            "You are an expert Python programmer. Solve the following problem.\n\n"
            + question.strip()
        )
        prompt_text += "\n[BEGIN]\n"
        ctx = (tokenizer.bos_token or "") + prompt_text

        prefix_tokens = tokenizer(ctx, return_tensors="pt")["input_ids"].to(device)

        # Generate
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        is_seed = "SEED" in type(model).__name__
        is_layerskip_speculative = (
            "LayerSkip" in type(model).__name__
            and gen_kwargs.get("assistant_early_exit") is not None
        )

        if is_seed or is_layerskip_speculative:
            sample, (generated_tokens, accepted_tokens), (
                accepted_lengths,
                accept_counts,
            ), _ = model.generate(
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

        # Decode generated code
        generated_text_raw = tokenizer.decode(
            sample[0, prefix_tokens.shape[1] :], skip_special_tokens=False
        )
        # Strip at EOS or common stop sequences.
        # Keep `if __name__ == "__main__"` for Online Judge style tasks,
        # otherwise stdio programs may never execute and produce empty output.
        stop_sequences = [tokenizer.eos_token]
        style_normalized = style.strip().lower().replace(" ", "_")
        stop_sequences.extend(["\n[END]", "\n[DONE]", "\n```", "\nclass "])
        if style_normalized != "online_judge":
            stop_sequences.append("\nif __name__")

        for stop_seq in stop_sequences:
            if stop_seq:
                generated_text_raw = generated_text_raw.split(stop_seq)[0]

        generated_text = _normalize_generated_code(generated_text_raw)

        # Run KodCode test cases on raw output and normalized output
        # Use style-aware test runner: stdio for Online Judge, pytest otherwise
        if style_normalized == "online_judge":
            _run_tests = _run_kodcode_stdio_tests
        else:
            _run_tests = _run_kodcode_tests

        raw_test_results = _run_tests(generated_text_raw, test_code, timeout=10)
        test_results = raw_test_results
        if generated_text != generated_text_raw and not raw_test_results["all_passed"]:
            normalized_test_results = _run_tests(generated_text, test_code, timeout=10)
            if normalized_test_results["all_passed"]:
                recovered_by_normalization += 1
            test_results = normalized_test_results

        all_passed = test_results["all_passed"]
        all_passed_raw = raw_test_results["all_passed"]

        if all_passed:
            total_correct += 1
        if all_passed_raw:
            total_correct_raw += 1
        total_problems += 1

        results_per_problem.append(
            {
                "question_id": example.get("conversation_id", idx),
                "difficulty": example.get("gpt_difficulty", "unknown"),
                "prompt": prompt_text,
                "generated_code_raw": generated_text_raw,
                "generated_code": generated_text,
                "reference_solution": ref_solution,
                "raw_test_results": raw_test_results,
                "test_results": test_results,
                "all_passed": all_passed,
                "all_passed_raw": all_passed_raw,
                "recovered_by_normalization": (not all_passed_raw and all_passed),
                "num_tests": test_results["total"],
                "num_passed": test_results["passed"],
            }
        )

        # Print progress
        accuracy = total_correct / total_problems
        if idx >= warmup and tputs:
            print(
                f"\n[{idx + 1}/{len(kodcode_test)}] "
                f"Accuracy: {total_correct}/{total_problems} = {accuracy:.2%} | "
                f"Thput: {np.mean(tputs):.2f} +/- {np.std(tputs):.2f} tok/s"
            )
        else:
            print(
                f"\n[{idx + 1}/{len(kodcode_test)}] "
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
        print(f"  Prompt: {ctx}")
        print(f"  Generated: {generated_text}")
        print(
            f"  Passed: {test_results['passed']}/{test_results['total']}"
        )

    # --- Aggregate metrics ---
    final_accuracy = (
        total_correct / total_problems if total_problems > 0 else 0.0
    )
    final_accuracy_raw = (
        total_correct_raw / total_problems if total_problems > 0 else 0.0
    )

    # Compute per-difficulty accuracy
    difficulty_stats = {}
    for r in results_per_problem:
        d = r["difficulty"]
        if d not in difficulty_stats:
            difficulty_stats[d] = {"correct": 0, "total": 0}
        difficulty_stats[d]["total"] += 1
        if r["all_passed"]:
            difficulty_stats[d]["correct"] += 1

    metrics = {
        "overall_accuracy": final_accuracy,
        "overall_accuracy_raw": final_accuracy_raw,
        "recovered_by_normalization": recovered_by_normalization,
        "total_correct": total_correct,
        "total_correct_raw": total_correct_raw,
        "total_problems": total_problems,
        "per_difficulty": {
            d: {
                "accuracy": (
                    s["correct"] / s["total"] if s["total"] > 0 else 0.0
                ),
                "correct": s["correct"],
                "total": s["total"],
            }
            for d, s in difficulty_stats.items()
        },
    }
    if tputs:
        metrics["throughput_mean_tok_s"] = float(np.mean(tputs))
        metrics["throughput_std_tok_s"] = float(np.std(tputs))
    if total_generated_tokens > 0:
        metrics["acceptance_rate"] = (
            total_accepted_tokens / total_generated_tokens
        )
    if total_accept_counts > 0:
        metrics["average_accepted_length"] = float(
            np.sum(total_accepted_lengths) / total_accept_counts
        )

    # --- Save results ---
    with open(os.path.join(output_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(output_path, "results.json"), "w") as f:
        json.dump(results_per_problem, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("KODCODE EVALUATION RESULTS")
    print("=" * 60)
    print(
        f"Overall Accuracy (strict): {final_accuracy:.2%} "
        f"({total_correct}/{total_problems})"
    )
    print(
        f"Raw Accuracy (no extraction): {final_accuracy_raw:.2%} "
        f"({total_correct_raw}/{total_problems})"
    )
    print(f"Recovered by extraction: {recovered_by_normalization}")
    for d, s in sorted(difficulty_stats.items()):
        d_acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        print(f"  {d}: {d_acc:.2%} ({s['correct']}/{s['total']})")
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
