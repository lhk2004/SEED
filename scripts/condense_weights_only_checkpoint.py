#!/usr/bin/env python3
"""Condense local checkpoints into a compact weights-only file.

This script keeps the output filename compatible with existing eval scripts
that look for files like:
  best-rank0_ema_weights_only.pt

It supports two input cases:
1) A full Composer checkpoint exists (e.g., best-rank0.pt): extract model/EMA
   weights, strip known prefixes, cast floating tensors to target dtype, save.
2) A weights-only checkpoint already exists: load it, cast floating tensors to
   target dtype, and overwrite in place.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import torch


DTYPE_MAP = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def _replace_in_state_dict_if_present(
    state_dict: dict[str, Any],
    prefix: str,
    replacement: str = "",
) -> None:
    keys = list(state_dict.keys())
    for key in keys:
        if prefix in key:
            new_key = key.replace(prefix, replacement)
            state_dict[new_key] = state_dict.pop(key)

    if hasattr(state_dict, "_metadata"):
        metadata_keys = list(state_dict._metadata.keys())
        for key in metadata_keys:
            if len(key) == 0:
                continue
            if key == prefix.replace(".", "") or prefix in key:
                new_key = key.replace(prefix, replacement)
                state_dict._metadata[new_key] = state_dict._metadata.pop(key)


def resolve_model_dir(model_folder: str, outputs_root: str) -> Path:
    model_path = Path(model_folder).expanduser()
    if not model_path.is_absolute():
        model_path = Path(outputs_root).expanduser() / model_path

    model_path = model_path.resolve()

    if model_path.name == "checkpoints" and model_path.is_dir():
        return model_path.parent

    if (model_path / "checkpoints").is_dir():
        return model_path

    raise FileNotFoundError(
        "Could not find a model directory with a checkpoints/ subdirectory at "
        f"{model_path}"
    )


def get_weights_only_path(model_dir: Path, ckpt_file: str, use_ema: bool) -> Path:
    base_name = ckpt_file[:-3] if ckpt_file.endswith(".pt") else ckpt_file
    ema_suffix = "_ema" if use_ema else ""
    return model_dir / "checkpoints" / f"{base_name}{ema_suffix}_weights_only.pt"


def get_weights_only_candidates(model_dir: Path, ckpt_file: str) -> list[Path]:
    base_name = ckpt_file[:-3] if ckpt_file.endswith(".pt") else ckpt_file
    checkpoints_dir = model_dir / "checkpoints"
    return [
        checkpoints_dir / f"{base_name}_ema_weights_only.pt",
        checkpoints_dir / f"{base_name}_weights_only.pt",
    ]


def extract_state_dict_from_full_checkpoint(
    full_ckpt_path: Path,
    load_ema_weights: bool,
) -> dict[str, Any]:
    ckpt = torch.load(
        full_ckpt_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )

    if "state" not in ckpt:
        raise KeyError(f"Expected key 'state' in full checkpoint: {full_ckpt_path}")

    if load_ema_weights:
        state_dict = None
        for alg_name, alg_state in ckpt["state"].get("algorithms", []):
            if alg_name == "EMA":
                state_dict = alg_state["ema_model"]["named_parameters_dict"]
                break
        if state_dict is None:
            raise ValueError("EMA weights not found in full checkpoint.")
    else:
        state_dict = ckpt["state"]["model"]

    _replace_in_state_dict_if_present(state_dict, "_orig_mod.")
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "model.")
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        state_dict, "_fsdp_wrapped_module."
    )
    return state_dict


def extract_state_dict_from_loaded_object(
    loaded_obj: Any,
    load_ema_weights: bool,
) -> dict[str, Any]:
    if not isinstance(loaded_obj, dict):
        raise TypeError(
            "Expected a dict checkpoint object; got "
            f"{type(loaded_obj).__name__}."
        )

    # Already a weights-only dict: all values are tensors (or tensor-like entries).
    if loaded_obj and all(torch.is_tensor(v) for v in loaded_obj.values()):
        return loaded_obj

    # Full Composer checkpoint mistakenly saved under weights-only filename.
    if "state" in loaded_obj:
        if load_ema_weights:
            state_dict = None
            for alg_name, alg_state in loaded_obj["state"].get("algorithms", []):
                if alg_name == "EMA":
                    state_dict = alg_state["ema_model"]["named_parameters_dict"]
                    break
            if state_dict is None:
                raise ValueError("EMA weights not found in loaded checkpoint object.")
        else:
            state_dict = loaded_obj["state"]["model"]

        _replace_in_state_dict_if_present(state_dict, "_orig_mod.")
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict, "module."
        )
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict, "model."
        )
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict, "_fsdp_wrapped_module."
        )
        return state_dict

    raise ValueError(
        "Unable to interpret checkpoint object. It is neither a weights-only dict "
        "nor a full checkpoint with a top-level 'state' key."
    )


def condense_state_dict_in_place(
    state_dict: dict[str, Any],
    target_dtype: torch.dtype,
) -> dict[str, Any]:
    dtype_before = Counter()
    dtype_after = Counter()
    total_tensors = 0
    converted_tensors = 0
    bytes_before = 0
    bytes_after = 0

    for key, value in list(state_dict.items()):
        if not torch.is_tensor(value):
            continue

        total_tensors += 1
        dtype_before[str(value.dtype)] += value.numel()
        bytes_before += value.numel() * value.element_size()

        if value.is_floating_point() and value.dtype != target_dtype:
            value = value.to(dtype=target_dtype)
            state_dict[key] = value
            converted_tensors += 1

        dtype_after[str(value.dtype)] += value.numel()
        bytes_after += value.numel() * value.element_size()

    print(f"Total tensors: {total_tensors}")
    print(f"Converted floating tensors: {converted_tensors}")
    print(f"Tensor payload before: {format_bytes(bytes_before)}")
    print(f"Tensor payload after:  {format_bytes(bytes_after)}")
    print(f"Dtype distribution before (numel): {dict(dtype_before)}")
    print(f"Dtype distribution after  (numel): {dict(dtype_after)}")

    return state_dict


def atomic_save(state_dict: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(destination.name + ".tmp")
    torch.save(state_dict, tmp_path)
    tmp_path.replace(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Condense a checkpoint to a compact weights-only file while keeping "
            "the eval-compatible filename."
        )
    )
    parser.add_argument(
        "model_folder",
        help=(
            "Model folder path (absolute) or model folder name under --outputs-root. "
            "You may also pass a checkpoints directory."
        ),
    )
    parser.add_argument(
        "--outputs-root",
        default=str(Path(__file__).resolve().parents[1] / "outputs"),
        help="Base outputs directory when model_folder is not an absolute path.",
    )
    parser.add_argument(
        "--ckpt-file",
        default="best-rank0.pt",
        help="Full checkpoint filename to extract from if needed.",
    )
    parser.add_argument(
        "--load-ema-weights",
        dest="load_ema_weights",
        action="store_true",
        help="Use EMA weights (default).",
    )
    parser.add_argument(
        "--no-load-ema-weights",
        dest="load_ema_weights",
        action="store_false",
        help="Use non-EMA model weights.",
    )
    parser.set_defaults(load_ema_weights=True)
    parser.add_argument(
        "--target-dtype",
        default="bf16",
        choices=sorted(DTYPE_MAP.keys()),
        help="Target dtype for floating tensors.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and convert in memory, but do not write output.",
    )
    parser.add_argument(
        "--force-target-filename",
        action="store_true",
        help=(
            "Always write to the target filename implied by --load-ema-weights "
            "(e.g., best-rank0_ema_weights_only.pt). By default, if an existing "
            "weights-only file is found, its original filename is preserved."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dtype = DTYPE_MAP[args.target_dtype]

    model_dir = resolve_model_dir(args.model_folder, args.outputs_root)
    checkpoints_dir = model_dir / "checkpoints"
    target_weights_only_path = get_weights_only_path(
        model_dir=model_dir,
        ckpt_file=args.ckpt_file,
        use_ema=args.load_ema_weights,
    )
    candidate_weights_only_paths = get_weights_only_candidates(
        model_dir=model_dir,
        ckpt_file=args.ckpt_file,
    )
    full_ckpt_path = checkpoints_dir / args.ckpt_file

    print(f"Model dir: {model_dir}")
    print(f"Target weights-only path: {target_weights_only_path}")
    print(f"Target dtype: {target_dtype}")

    existing_weights_only_path = next(
        (p for p in candidate_weights_only_paths if p.exists()),
        None,
    )

    if existing_weights_only_path is not None:
        print(f"Loading existing weights-only checkpoint: {existing_weights_only_path}")
        loaded_obj = torch.load(
            existing_weights_only_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = extract_state_dict_from_loaded_object(
            loaded_obj,
            load_ema_weights=args.load_ema_weights,
        )
        if args.force_target_filename:
            output_weights_only_path = target_weights_only_path
        else:
            output_weights_only_path = existing_weights_only_path
    elif full_ckpt_path.exists():
        print(f"Loading full checkpoint: {full_ckpt_path}")
        state_dict = extract_state_dict_from_full_checkpoint(
            full_ckpt_path=full_ckpt_path,
            load_ema_weights=args.load_ema_weights,
        )
        output_weights_only_path = target_weights_only_path
    else:
        raise FileNotFoundError(
            "Neither weights-only nor full checkpoint found. Looked for:\n"
            + "\n".join([f"- {p}" for p in candidate_weights_only_paths])
            + "\n"
            f"- {full_ckpt_path}"
        )

    if output_weights_only_path != target_weights_only_path:
        print(
            "Preserving existing filename: "
            f"{output_weights_only_path.name} (instead of {target_weights_only_path.name})"
        )

    condense_state_dict_in_place(state_dict, target_dtype=target_dtype)

    if args.dry_run:
        print("Dry run enabled; no file written.")
        return

    atomic_save(state_dict, output_weights_only_path)
    print(f"Wrote condensed checkpoint: {output_weights_only_path}")
    print(f"On-disk file size: {format_bytes(output_weights_only_path.stat().st_size)}")


if __name__ == "__main__":
    main()
