"""
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
"""

import json
import os
import re
import sys
from typing import Any, List, Tuple

import accelerate
import hydra
import numpy as np
import torch
from lm_eval.api.model import LM
from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.utils import make_table
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    PreTrainedTokenizer,
)

from datasets import Dataset
from scripts.eval.gsm8k_metrics import (
    boxed_answer_accuracy,
    prepare_gsm8k_response,
    truncate_response,
)
from scripts.utils import (
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
    format_number,
    count_parameters,
    set_seed,
)
from src.utils import fsspec_exists, fsspec_mkdirs


class LMEvalHarnessModel(LM):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        generated_samples_output_path: str,
        tokenizer: PreTrainedTokenizer,
        pretrained_model_revision: str | None = None,
        load_ema_weights: bool = False,
        ckpt_file: str = "best-rank0.pt",  # best-rank0.pt or latest-rank0.pt
        gen_kwargs: Any | None = None,
        accelerator: accelerate.Accelerator | None = None,
        throughput_run: bool = False,
        throughput_samples: int = 100,
        throughput_warmup: int = 100,
        model_config_overrides: dict[str, Any] | None = None,
    ):
        """
        Args:
            pretrained_model_name_or_path (str): Path to ckpt dir or HF model repo.
            generated_samples_output_path (str): Path to generated samples dir.
            tokenizer (str): Tokenizer name or path.
            pretrained_model_revision (Optional[str]): Revision (e.g., commit id)
                passed to `.from_pretrained` model instantiation.
            load_ema_weights (bool): Whether to load ema weights (for local ckpts).
            ckpt_file (str): Name of ckpt file (for local ckpts).
            gen_kwargs (dict): Generator kwargs.
                Ideally this should be passed via `lm_eval.evaluator.simple_evaluate`,
                however this method expects `gen_kwargs` as string with comma-separated
                arguments, which is not compatible in our hydra framework.
            throughput_run (bool): Whether to run the evaluation throughput.
            model_config_overrides (dict[str, Any]): Model config overrides.
        """
        if "fsdp" in pretrained_model_name_or_path:
            load_ema_weights = False
        # SEED and AR trained on GSM8K still used ema
        # if ("seed" in pretrained_model_name_or_path and "e2d2" not in pretrained_model_name_or_path) and "gsm8k" in pretrained_model_name_or_path and "fsdp" in pretrained_model_name_or_path:
        #     load_ema_weights = True
        super().__init__()
        self.generated_samples_output_path = generated_samples_output_path
        if not fsspec_exists(self.generated_samples_output_path):
            fsspec_mkdirs(self.generated_samples_output_path)
        self.accelerator = accelerator
        if self.accelerator is not None:
            device = self.accelerator.device
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._rank = 0
            self._world_size = 1
        self.device = torch.device(f"{device}")

        model_config_overrides = (
            {} if model_config_overrides is None else model_config_overrides
        )
        self._is_hf_model = not fsspec_exists(
            os.path.join(pretrained_model_name_or_path, "config.yaml")
        )
        if not self._is_hf_model:
            model = load_model_from_ckpt_dir_path(
                path_to_ckpt_dir=pretrained_model_name_or_path,
                load_ema_weights=load_ema_weights,
                ckpt_file=ckpt_file,
                device=self.device,
                **model_config_overrides,
            )
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    revision=pretrained_model_revision,
                    **model_config_overrides,
                )
            except:  # Model not compatible with CausalLM
                model = AutoModelForMaskedLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    revision=pretrained_model_revision,
                    **model_config_overrides,
                )
        self.model = model.to(self.device)
        print(f"Num. params: {format_number(count_parameters(model, trainable=False))}")
        print(f"Num. trainable params: {format_number(count_parameters(model))}")
        self.model.eval()
        self.tokenizer = maybe_add_missing_special_tokens(tokenizer)
        # print tokenizer name
        print(f"Using tokenizer: {self.tokenizer.name_or_path}")
        self.gen_kwargs = gen_kwargs
        self.throughput_run = throughput_run
        self.throughput_warmup = throughput_warmup
        self.throughput_samples = throughput_samples
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def loglikelihood_rolling(self, requests) -> List[float]:
        raise NotImplementedError

    def generate_until(self, requests, **generation_kwargs):
        # Detect task type from the first request
        is_gsm8k = (
            len(requests) > 0
            and hasattr(requests[0], "doc")
            and "answer" in requests[0].doc
        )

        # TODO: Move this to utils file
        def _tokenize_gsm8k(
            e,
            prefix_text: str = (
                "Please reason step by step, and put your final answer within "
                + "$\\boxed{}$."
            ),
        ):
            ctx = e["prefix"]
            ctx = re.sub(
                r"^####\s*(\d+)\s*$",
                r"$\\boxed{\1}$" + (self.tokenizer.eos_token or ""),
                ctx,
                flags=re.MULTILINE,
            )

            bos = self.tokenizer.bos_token or ""
            eos = self.tokenizer.eos_token or ""
            ctx = ctx.replace("Question: ", f"{bos}{prefix_text} ")
            if "SEED" in type(self.model).__name__:
                ctx = ctx.replace("\nAnswer:", f"{eos}")
            else:
                ctx = ctx.replace("\nAnswer:", f"{eos}Answer:")
            prefix_tokens = self.tokenizer(ctx)["input_ids"]
            return {
                "prefix_text": ctx,
                "prefix": prefix_tokens,
                "target": e["target"],
            }

        def _tokenize_default(e):
            ctx = e["prefix"]
            ctx = self.tokenizer.bos_token + ctx
            prefix_tokens = self.tokenizer(ctx)["input_ids"]
            return {
                "prefix_text": ctx,
                "prefix": prefix_tokens,
                "target": e["target"],
            }

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        if is_gsm8k:
            ds = ds.map(_tokenize_gsm8k)
        else:
            ds = ds.map(_tokenize_default)
        ds = ds.with_format("torch")
        res = []
        res_for_json = []
        correct, total = 0, 0
        tputs = []
        total_generated_tokens = 0
        total_accepted_tokens = 0
        total_accepted_lengths = []
        total_accept_counts = 0
        total_draft_lengths = []
        total_drafting_time_s = 0.0
        total_all_time_s = 0.0
        total_draft_position_attempt_counts: List[int] = []
        total_draft_position_accept_counts: List[int] = []
        for i, elem in tqdm(
            enumerate(ds), desc="Generating", total=len(ds), disable=(self.rank != 0)
        ):
            if (
                self.throughput_run
                and i >= self.throughput_samples + self.throughput_warmup
            ):
                tputs_path = (
                    f"{self.generated_samples_output_path}/throughput-rank{self.rank}"
                )
                with open(f"{tputs_path}.json", "w") as f:
                    json.dump(
                        {
                            "throughput_mean": np.mean(tputs),
                            "throughput_std": np.std(tputs),
                            "throughput_all": tputs,
                        },
                        f,  # type: ignore
                        indent=2,
                    )
                sys.exit(0)
            if self.rank == 0:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_event, end_event = None, None
            inputs = elem["prefix"][None, ...].to(self.device)
            generate_kwargs = dict(self.gen_kwargs)
            if self._is_hf_model:
                # HF validates model kwargs and does not accept disable_pbar.
                generate_kwargs["attention_mask"] = torch.ones_like(inputs)
            else:
                generate_kwargs["disable_pbar"] = self.rank != 0
            sample_output = self.model.generate(inputs=inputs, **generate_kwargs)

            if (
                isinstance(sample_output, tuple)
                and len(sample_output) >= 3
                and isinstance(sample_output[1], tuple)
                and isinstance(sample_output[2], tuple)
            ):
                if len(sample_output) == 4:
                    sample, (generated_tokens, accepted_tokens), (accepted_lengths, accept_counts), (drafting_time_s, all_time_s) = sample_output
                    total_drafting_time_s += drafting_time_s
                    total_all_time_s += all_time_s
                else:
                    sample, (generated_tokens, accepted_tokens), (accepted_lengths, accept_counts) = sample_output
                    drafting_time_s, all_time_s = 0.0, 0.0
                draft_position_stats = getattr(self.model, "_last_draft_position_acceptance", None)
            else:
                sample = sample_output
                generated_tokens = int(sample.shape[-1] - elem["prefix"].numel())
                accepted_tokens = generated_tokens
                # Plain autoregressive generation accepts one token per step.
                accepted_lengths = [1] * generated_tokens
                accept_counts = generated_tokens
                draft_position_stats = None
                drafting_time_s, all_time_s = 0.0, 0.0

            if self.rank == 0 and all_time_s > 0:
                print(f"[TIME STATS] Sample Drafting Time: {drafting_time_s:.4f}s / All Time: {all_time_s:.4f}s ({drafting_time_s/all_time_s:.2%})")
                print(f"[TIME STATS] Total Drafting Time: {total_drafting_time_s:.4f}s / Total All Time: {total_all_time_s:.4f}s ({total_drafting_time_s/total_all_time_s:.2%})")

            total_generated_tokens += generated_tokens
            total_accepted_tokens += accepted_tokens
            total_accepted_lengths.extend(accepted_lengths)
            total_accept_counts += accept_counts
            if draft_position_stats is not None:
                total_draft_lengths.extend(
                    int(length)
                    for length in draft_position_stats.get("draft_lengths", [])
                )
                attempt_counts = draft_position_stats.get("attempt_counts", [])
                accept_counts_pos = draft_position_stats.get("accept_counts", [])
                max_len = max(
                    len(total_draft_position_attempt_counts),
                    len(attempt_counts),
                    len(total_draft_position_accept_counts),
                    len(accept_counts_pos),
                )
                if len(total_draft_position_attempt_counts) < max_len:
                    total_draft_position_attempt_counts.extend(
                        [0] * (max_len - len(total_draft_position_attempt_counts))
                    )
                if len(total_draft_position_accept_counts) < max_len:
                    total_draft_position_accept_counts.extend(
                        [0] * (max_len - len(total_draft_position_accept_counts))
                    )
                for idx, val in enumerate(attempt_counts):
                    total_draft_position_attempt_counts[idx] += int(val)
                for idx, val in enumerate(accept_counts_pos):
                    total_draft_position_accept_counts[idx] += int(val)
            if self.rank == 0:
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time_s = start_event.elapsed_time(end_event) / 1000
                tput = (sample.numel() - elem["prefix"].numel()) / elapsed_time_s
                if i >= self.throughput_warmup:
                    tputs.append(tput)
            result = self.tokenizer.decode(sample[0, len(elem["prefix"]) :])
            result = truncate_response(
                result, elem["target"]["until"], self.tokenizer.eos_token
            )

            result, predicted_ans = prepare_gsm8k_response(result)
            is_correct = boxed_answer_accuracy(
                predicted_ans, requests[i].doc["answer"]
            )
            if self.rank == 0:
                print("=" * 20)
                print("Prefix:", elem["prefix_text"])
                print("Generated:", result)
                print("(Ground truth):", requests[i].doc["answer"])
                print("=" * 20, end="\n\n")
            res.append(result)

            # log accuracy
            correct += int(is_correct)
            total += 1
            res_for_json.append(
                {
                    "prefix": elem["prefix_text"],
                    "result": result,
                }
            )
            # torch.cuda.empty_cache()
            if self.rank == 0:
                if is_gsm8k:
                    print(f"\nAccuracy: {correct}/{total} = {correct / total:.2%}\n")
                else:
                    print(f"\nCompleted: {total}/{len(ds)}\n")
                acceptance_rate = (
                    total_accepted_tokens / total_generated_tokens
                    if total_generated_tokens > 0
                    else 0.0
                )
                avg_accepted_len = (
                    np.sum(total_accepted_lengths) / total_accept_counts
                    if total_accept_counts > 0
                    else 0.0
                )
                avg_draft_len = (
                    np.mean(total_draft_lengths)
                    if len(total_draft_lengths) > 0
                    else 0.0
                )
                print(f"Total generated tokens: {total_generated_tokens}, Total accepted tokens: {total_accepted_tokens}, Acceptance rate: {acceptance_rate:.2%}")
                print(f"Average accepted length: {avg_accepted_len:.2f}")
                if len(total_draft_lengths) > 0:
                    print(f"Running avg draft length: {avg_draft_len:.2f}")
                if len(total_draft_position_attempt_counts) > 0:
                    per_pos_strings = []
                    for pos, (acc, att) in enumerate(
                        zip(total_draft_position_accept_counts, total_draft_position_attempt_counts),
                        start=1,
                    ):
                        rate = (acc / att) if att > 0 else 0.0
                        per_pos_strings.append(f"p{pos}:{rate:.2%} ({acc}/{att})")
                    print("Per-position acceptance rate: " + ", ".join(per_pos_strings))
                if i >= self.throughput_warmup:
                    print(
                        f"Thput (tok/s): {np.mean(tputs):0.2f} +/- {np.std(tputs):0.2f}"
                    )
                else:
                    print(f"Thput (tok/s): {tput:0.2f}")

        if self.rank == 0 and len(total_draft_position_attempt_counts) > 0:
            per_position_summary = []
            for pos, (acc, att) in enumerate(
                zip(total_draft_position_accept_counts, total_draft_position_attempt_counts),
                start=1,
            ):
                per_position_summary.append(
                    {
                        "position": pos,
                        "accept_count": int(acc),
                        "attempt_count": int(att),
                        "acceptance_rate": float(acc / att) if att > 0 else 0.0,
                    }
                )
            draft_position_metrics_path = (
                f"{self.generated_samples_output_path}/draft_position_acceptance-rank{self.rank}.json"
            )
            with open(draft_position_metrics_path, "w") as f:
                json.dump(per_position_summary, f, indent=2)

            draft_position_metrics_txt_path = (
                f"{self.generated_samples_output_path}/draft_position_acceptance-rank{self.rank}.txt"
            )
            with open(draft_position_metrics_txt_path, "w") as f:
                for row in per_position_summary:
                    f.write(
                        f"position={row['position']}, acceptance_rate={row['acceptance_rate']:.6f}, "
                        f"accept_count={row['accept_count']}, attempt_count={row['attempt_count']}\n"
                    )

        samples_path = f"{self.generated_samples_output_path}/rank{self.rank}"
        with open(f"{samples_path}.json", "w") as f:
            json.dump(
                res_for_json,
                f,  # type: ignore
                indent=2,
            )
        print(f"RANK {self.rank} completed!")
        return res


@hydra.main(version_base=None, config_path="../../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    accelerator = accelerate.Accelerator()
    accelerator = accelerate.Accelerator() if accelerator.num_processes > 1 else None
    set_seed(cfg.seed)
    model = hydra.utils.instantiate(cfg.task.model, accelerator=accelerator)
    evaluation_tracker = EvaluationTracker(output_path=cfg.output_path)
    # A custom LM instance bypasses lm-eval's model_args setup. Without this
    # metadata the tracker tries to join the output path with a None model name
    # and silently skips both aggregated and per-sample JSON reports.
    evaluation_tracker.general_config_tracker.log_experiment_args(
        model_source=type(model).__name__,
        model_args=f"pretrained={cfg.pretrained_model_name_or_path}",
        system_instruction=None,
        chat_template=None,
        fewshot_as_multiturn=False,
    )
    results = hydra.utils.call(cfg.task, model=model)
    if results is not None and (
        accelerator is None or accelerator.local_process_index == 0
    ):
        if isinstance(results, DictConfig):
            results = OmegaConf.to_container(results, resolve=True)
        if not isinstance(results, dict):
            raise RuntimeError(
                f"Unexpected lm-eval return type: {type(results)}. "
                "Expected a dict-like object."
            )

        # Some lm-eval versions/tasks may not return per-sample logs.
        try:
            samples = results.pop("samples")
        except Exception:
            samples = None

        has_standard_schema = "results" in results and "configs" in results
        if has_standard_schema:
            evaluation_tracker.save_results_aggregated(
                results=results,
                samples=samples if samples is not None else {},
            )
            if samples is not None:
                for task_name, config in results["configs"].items():
                    if task_name in samples:
                        evaluation_tracker.save_results_samples(
                            task_name=task_name,
                            samples=samples[task_name],
                        )
            print(make_table(results))
            if "groups" in results:
                print(make_table(results, "groups"))
        else:
            print("Warning: lm-eval returned a non-standard result schema.")
            print(json.dumps(results, indent=2, default=str))

        metrics_f = f"{cfg.task.model.generated_samples_output_path}/metrics.txt"
        with open(metrics_f, "w") as f:
            if has_standard_schema:
                f.write(make_table(results))
            else:
                f.write(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    register_useful_resolvers()
    main()
