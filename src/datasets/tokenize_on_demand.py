import random
import re
from typing import Any, Dict, Literal

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer

from datasets import load_dataset

_QUESTION_PREFIX = (
    "Please reason step by step, and put your final answer within $\\boxed{}$. "
)
_SUMMARY_PREFIX = "Summarize the following article: "


class GSM8KDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: Literal["train", "test"],
        max_length: int,
        dataset_path: str = "openai/gsm8k",
        config_name: Literal["main", "socratic"] = "main",
        padding: bool = False,
        add_special_tokens: bool = True,
        source_prompt_text: str | None = _QUESTION_PREFIX,
        target_prompt_text: str | None = "Answer: ",
        source_key: str = "question",
        target_key: str = "answer",
        num_shot: int = 0,
        # Unused tokenizer arg (compat. with other dataset loading functions/classes)
        **_: Dict[str, Any],
    ):
        self.tokenizer = tokenizer
        self.split = split
        self.dataset = load_dataset(
            dataset_path, config_name, split=split, trust_remote_code=True
        )
        self.max_length = max_length
        self.padding = padding
        self.add_special_tokens = add_special_tokens
        self.source_prompt_text = source_prompt_text
        self.target_prompt_text = target_prompt_text
        self.source_key = source_key
        self.target_key = target_key
        self.num_shot = num_shot
        self._arange = range(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def _few_shot_idxs(self, exclude: int):
        candidates = [x for x in self._arange if x != exclude]
        if self.split == "train":
            return random.sample(candidates, self.num_shot)
        return [(exclude + n) % len(self.dataset) for n in range(self.num_shot)]

    def __getitem__(self, idx):
        example = self.dataset[idx]
        sp = (self.tokenizer.bos_token if self.add_special_tokens else "") + (
            self.source_prompt_text if self.source_prompt_text is not None else ""
        )
        tp = self.target_prompt_text if self.target_prompt_text is not None else ""
        if self.num_shot > 0:
            example_shots = [self.dataset[fsi] for fsi in self._few_shot_idxs(idx)]
            source = "\n".join(
                [
                    sp
                    + i[self.source_key]  # type: ignore
                    + (self.tokenizer.eos_token if self.add_special_tokens else "")
                    + tp
                    + re.sub(  # type: ignore
                        r"^####\s*(\d+)\s*$",
                        r"$\\boxed{\1}$",
                        i[self.target_key],
                        flags=re.MULTILINE,
                    )
                    + (self.tokenizer.eos_token if self.add_special_tokens else "")
                    for i in example_shots
                ]
            )
        else:
            source = ""
        source = (
            source
            + sp
            + example[self.source_key]  # type: ignore
            + (self.tokenizer.eos_token if self.add_special_tokens else "")
        )
        target = (
            tp
            + re.sub(  # type: ignore
                r"^####\s*(\d+)\s*$",
                r"$\\boxed{\1}$",
                example[self.target_key],
                flags=re.MULTILINE,
            )
            + (self.tokenizer.eos_token if self.add_special_tokens else "")
        )

        qa_tokenized = self.tokenizer.batch_encode_plus(
            [source, target],
            max_length=self.max_length // 2,
            padding=self.padding,
            add_special_tokens=False,  # (potentially) added manually, above
            truncation=True,
        )

        input_ids = torch.cat(
            [torch.LongTensor(t) for t in qa_tokenized["input_ids"]], dim=-1
        )
        attention_mask = torch.cat(
            [torch.LongTensor(a) for a in qa_tokenized["attention_mask"]], dim=-1
        )
        context_mask = torch.cat(
            (
                torch.LongTensor(qa_tokenized["attention_mask"][0]),
                torch.zeros_like(torch.LongTensor(qa_tokenized["input_ids"][1])),
            ),
            dim=-1,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "context_mask": context_mask,
        }


_SCIENCEQA_ANSWER_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]
_SCIENCEQA_PREFIX = (
    "The following is a multiple choice question. Think step by step and then "
    "give your final answer.\n\n"
)


class ScienceQADataset(Dataset):
    """ScienceQA closed-choice, text-only dataset.

    Train split merges the HF train + validation splits.
    Test split uses the HF test split (capped at ``test_size`` samples).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: Literal["train", "test"] = "train",
        max_length: int = 768,
        dataset_path: str = "derek-thomas/ScienceQA",
        padding: bool = False,
        add_special_tokens: bool = True,
        num_shot: int = 0,
        test_size: int = 1000,
        sampling_seed: int = 42,
        source_prompt_text: str | None = _SCIENCEQA_PREFIX,
        target_prompt_text: str | None = "Answer: ",
        **_: Dict[str, Any],
    ):
        from datasets import concatenate_datasets

        self.tokenizer = tokenizer
        self.split = split
        self.source_prompt_text = source_prompt_text
        self.target_prompt_text = target_prompt_text

        def _is_text_only_closed_choice(example):
            return example["image"] is None and example["task"] == "closed choice"

        if split == "train":
            train_ds = load_dataset(dataset_path, split="train", trust_remote_code=True)
            val_ds = load_dataset(dataset_path, split="validation", trust_remote_code=True)
            combined = concatenate_datasets([train_ds, val_ds])
            combined = combined.filter(_is_text_only_closed_choice)
            self.dataset = combined
        else:
            test_ds = load_dataset(dataset_path, split="test", trust_remote_code=True)
            test_ds = test_ds.filter(_is_text_only_closed_choice)
            if len(test_ds) > test_size:
                rng = np.random.RandomState(sampling_seed)
                indices = sorted(
                    rng.choice(len(test_ds), size=test_size, replace=False).tolist()
                )
                test_ds = test_ds.select(indices)
            self.dataset = test_ds

        # Remove the image column to avoid serialisation issues
        if "image" in self.dataset.column_names:
            self.dataset = self.dataset.remove_columns("image")

        self.max_length = max_length
        self.padding = padding
        self.add_special_tokens = add_special_tokens
        self.num_shot = num_shot
        self._arange = range(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def _few_shot_idxs(self, exclude: int):
        candidates = [x for x in self._arange if x != exclude]
        if self.split == "train":
            return random.sample(candidates, self.num_shot)
        return [(exclude + n) % len(self.dataset) for n in range(self.num_shot)]

    @staticmethod
    def _format_question(question: str, choices: list[str], hint: str | None = None) -> str:
        parts = [question]
        if hint:
            parts.append(f"Hint: {hint}")
        for idx, choice in enumerate(choices):
            parts.append(f"({_SCIENCEQA_ANSWER_LETTERS[idx]}) {choice}")
        return "\n".join(parts)

    @staticmethod
    def _format_answer(solution: str, answer_idx: int) -> str:
        letter = _SCIENCEQA_ANSWER_LETTERS[answer_idx]
        return f"{solution}\nThe answer is ({letter})."

    @property
    def target_references(self) -> list[str]:
        return [
            _SCIENCEQA_ANSWER_LETTERS[ex["answer"]]
            for ex in self.dataset
        ]

    def __getitem__(self, idx):
        example = self.dataset[idx]
        sp = (self.tokenizer.bos_token if self.add_special_tokens else "") + (
            self.source_prompt_text if self.source_prompt_text is not None else ""
        )
        tp = self.target_prompt_text if self.target_prompt_text is not None else ""
        eos = self.tokenizer.eos_token if self.add_special_tokens else ""

        q_text = self._format_question(
            example["question"], example["choices"], example.get("hint", "")
        )

        if self.num_shot > 0:
            example_shots = [self.dataset[fsi] for fsi in self._few_shot_idxs(idx)]
            source = "\n".join(
                [
                    sp
                    + self._format_question(
                        i["question"], i["choices"], i.get("hint", "")
                    )
                    + eos
                    + tp
                    + self._format_answer(i["solution"], i["answer"])
                    + eos
                    for i in example_shots
                ]
            )
        else:
            source = ""

        source = source + sp + q_text + eos
        target = tp + self._format_answer(example["solution"], example["answer"]) + eos

        qa_tokenized = self.tokenizer.batch_encode_plus(
            [source, target],
            max_length=self.max_length // 2,
            padding=self.padding,
            add_special_tokens=False,
            truncation=True,
        )

        input_ids = torch.cat(
            [torch.LongTensor(t) for t in qa_tokenized["input_ids"]], dim=-1
        )
        attention_mask = torch.cat(
            [torch.LongTensor(a) for a in qa_tokenized["attention_mask"]], dim=-1
        )
        context_mask = torch.cat(
            (
                torch.LongTensor(qa_tokenized["attention_mask"][0]),
                torch.zeros_like(torch.LongTensor(qa_tokenized["input_ids"][1])),
            ),
            dim=-1,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "context_mask": context_mask,
        }


class CNNDailyMailDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: Literal["train", "validation", "test"],
        max_length: int,
        dataset_path: str = "abisee/cnn_dailymail",
        config_name: Literal["1.0.0", "2.0.0", "3.0.0"] = "3.0.0",
        padding: bool = False,
        add_special_tokens: bool = True,
        source_prompt_text: str | None = _SUMMARY_PREFIX,
        target_prompt_text: str | None = "Summary: ",
        source_key: str = "article",
        target_key: str = "highlights",
        separate_input_output: bool = False,
        truncate: bool = True,
        max_samples: int | None = None,
        filter_by_length: bool = False,
        source_max_length_ratio: float = 0.5,
        target_max_length_ratio: float = 0.5,
        filter_max_length: int | None = None,
        # Unused tokenizer arg (compat. with other dataset loading functions/classes)
        **_: Dict[str, Any],
    ):
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.source_prompt_text = source_prompt_text
        self.target_prompt_text = target_prompt_text

        self.dataset = load_dataset(
            dataset_path, config_name, split=split, trust_remote_code=True
        )

        self.source_max_length_ratio = source_max_length_ratio
        self.target_max_length_ratio = target_max_length_ratio

        # Filter samples by tokenized source/target length before applying max_samples
        if filter_by_length:
            filter_len = filter_max_length if filter_max_length is not None else max_length
            max_source_len = source_max_length_ratio * filter_len
            max_target_len = target_max_length_ratio * filter_len

            original_len = len(self.dataset)

            def _length_filter(examples):
                sources = [
                    self._format_source_text(str(s))
                    for s in examples[source_key]
                ]
                targets = [
                    self._format_target_text(str(t))
                    for t in examples[target_key]
                ]
                s_enc = tokenizer(
                    sources, add_special_tokens=False, truncation=False
                )
                t_enc = tokenizer(
                    targets, add_special_tokens=False, truncation=False
                )
                return [
                    len(s_ids) < max_source_len and len(t_ids) < max_target_len
                    for s_ids, t_ids in zip(
                        s_enc["input_ids"], t_enc["input_ids"]
                    )
                ]

            self.dataset = self.dataset.filter(
                _length_filter, batched=True, batch_size=1000
            )
            print(
                f"[CNNDailyMailDataset] Length filter ({split}): "
                f"{len(self.dataset)}/{original_len} samples passed "
                f"(source < {max_source_len:.0f} tokens, "
                f"target < {max_target_len:.0f} tokens)"
            )

        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        self.max_length = max_length
        self.padding = padding
        self.separate_input_output = separate_input_output
        self.source_key = source_key
        self.target_key = target_key
        self.truncate = truncate

    def _format_source_text(self, source: str) -> str:
        source = (self.source_prompt_text or "") + source
        if self.add_special_tokens:
            source = (
                (self.tokenizer.bos_token or "")
                + source
                + (self.tokenizer.eos_token or "")
            )
        return source

    def _format_target_text(self, target: str) -> str:
        if self.target_prompt_text is not None:
            target = self.target_prompt_text + target
        if self.add_special_tokens:
            target = target + (self.tokenizer.eos_token or "")
        return target

    @property
    def target_references(self) -> list[str]:
        """Helper method to retrieve list of ground truth labels for downstream eval."""
        return self.dataset[self.target_key]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        source = self._format_source_text(str(example[self.source_key]))
        target = self._format_target_text(str(example[self.target_key]))

        source_max_len = int(self.source_max_length_ratio * self.max_length)
        target_max_len = int(self.target_max_length_ratio * self.max_length)

        source_tokenized = self.tokenizer.encode_plus(
            source,
            max_length=source_max_len,
            padding=self.padding,
            add_special_tokens=False,  # (potentially) added manually, above
            truncation=self.truncate,
        )
        target_tokenized = self.tokenizer.encode_plus(
            target,
            max_length=target_max_len,
            padding=self.padding,
            add_special_tokens=False,  # (potentially) added manually, above
            truncation=self.truncate,
        )

        if self.separate_input_output:
            input_ids = torch.LongTensor(source_tokenized["input_ids"])
            attention_mask = torch.LongTensor(source_tokenized["attention_mask"])
            context_mask = torch.LongTensor(source_tokenized["attention_mask"])
            output_ids = torch.LongTensor(target_tokenized["input_ids"])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "context_mask": context_mask,
                "output_ids": output_ids,
            }
        else:
            input_ids = torch.cat(
                [
                    torch.LongTensor(source_tokenized["input_ids"]),
                    torch.LongTensor(target_tokenized["input_ids"]),
                ],
                dim=-1,
            )
            attention_mask = torch.cat(
                [
                    torch.LongTensor(source_tokenized["attention_mask"]),
                    torch.LongTensor(target_tokenized["attention_mask"]),
                ],
                dim=-1,
            )
            context_mask = torch.cat(
                (
                    torch.LongTensor(source_tokenized["attention_mask"]),
                    torch.zeros_like(
                        torch.LongTensor(target_tokenized["input_ids"])
                    ),
                ),
                dim=-1,
            )
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "context_mask": context_mask,
            }


class KodCodeDataset(Dataset):
    """KodCode dataset for code generation training and evaluation.

    Uses the KodCode/KodCode-V1-SFT-R1 dataset. We filter for version=="v1.1"
    samples, randomly sample half (with a fixed seed for reproducibility), then
    use the last ``test_size`` (default 1000) of those as the test set and the
    rest as the training set.

    Prompt format (aligned to eval):
      You are an expert Python programmer. Solve the following problem.

      {question}
      [BEGIN]

    Source is the prompt, target is the solution code.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: Literal["train", "test"] = "train",
        max_length: int = 1024,
        dataset_path: str = "KodCode/KodCode-V1-SFT-R1",
        padding: bool = False,
        add_special_tokens: bool = True,
        num_shot: int = 0,
        test_size: int = 1000,
        sampling_seed: int = 42,
        difficulty: str | None = None,
        # Unused tokenizer arg (compat. with other dataset loading functions/classes)
        **_: Dict[str, Any],
    ):
        self.tokenizer = tokenizer
        self.split = split
        full_dataset = load_dataset(dataset_path, split="train", trust_remote_code=True)
        # Filter to v1.1 samples only
        full_dataset = full_dataset.filter(
            lambda x: x.get("version", "") == "v1.1"
        )
        # Randomly sample half with a fixed seed for reproducibility
        rng = np.random.RandomState(sampling_seed)
        num_samples = len(full_dataset) // 2
        sampled_indices = sorted(
            rng.choice(len(full_dataset), size=num_samples, replace=False).tolist()
        )
        full_dataset = full_dataset.select(sampled_indices)
        # Split: last test_size as test, rest as train
        train_size = len(full_dataset) - test_size
        if split == "train":
            self.dataset = full_dataset.select(range(train_size))
        else:  # test / eval
            self.dataset = full_dataset.select(range(train_size, len(full_dataset)))
        # Optionally filter by difficulty
        if difficulty is not None:
            self.dataset = self.dataset.filter(
                lambda x: x.get("gpt_difficulty", "") == difficulty
            )
        self.max_length = max_length
        self.padding = padding
        self.add_special_tokens = add_special_tokens
        self.num_shot = num_shot
        self._arange = range(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def _few_shot_idxs(self, exclude: int):
        candidates = [x for x in self._arange if x != exclude]
        if self.split == "train":
            return random.sample(candidates, self.num_shot)
        return [(exclude + n) % len(self.dataset) for n in range(self.num_shot)]

    @staticmethod
    def _build_prompt(question: str) -> str:
        """Build KodCode evaluation prompt."""
        prompt = (
            "You are an expert Python programmer. Solve the following problem.\n\n"
            + question.strip()
        )
        prompt += "\n[BEGIN]\n"
        return prompt

    def __getitem__(self, idx):
        example = self.dataset[idx]
        bos = (self.tokenizer.bos_token or "") if self.add_special_tokens else ""

        question = example["question"]
        solution = example["solution"]

        if self.num_shot > 0:
            example_shots = [self.dataset[fsi] for fsi in self._few_shot_idxs(idx)]
            source = "\n".join(
                [
                    bos
                    + self._build_prompt(i["question"])
                    + i["solution"]
                    + (self.tokenizer.eos_token if self.add_special_tokens else "")
                    for i in example_shots
                ]
            )
        else:
            source = ""

        source = source + bos + self._build_prompt(question)
        target = (
            solution
            + (self.tokenizer.eos_token if self.add_special_tokens else "")
        )

        qa_tokenized = self.tokenizer.batch_encode_plus(
            [source, target],
            max_length=self.max_length // 2,
            padding=self.padding,
            add_special_tokens=False,
            truncation=True,
        )

        input_ids = torch.cat(
            [torch.LongTensor(t) for t in qa_tokenized["input_ids"]], dim=-1
        )
        attention_mask = torch.cat(
            [torch.LongTensor(a) for a in qa_tokenized["attention_mask"]], dim=-1
        )
        context_mask = torch.cat(
            (
                torch.LongTensor(qa_tokenized["attention_mask"][0]),
                torch.zeros_like(torch.LongTensor(qa_tokenized["input_ids"][1])),
            ),
            dim=-1,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "context_mask": context_mask,
        }
