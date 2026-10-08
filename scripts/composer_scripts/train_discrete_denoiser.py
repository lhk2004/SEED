import logging

# import time  # Use for multi-node
import hydra
import torch
import torch.distributed as torch_dist
from composer.models import HuggingFaceModel
from composer.utils import dist, reproducibility
from omegaconf import DictConfig, OmegaConf, open_dict

from scripts.utils import (
    count_parameters,
    format_number,
    maybe_add_missing_special_tokens,
    print_and_save_config,
    register_useful_resolvers,
)
from src.backbone.encoder_decoder import LLMasEncoderDecoderShareKVEncoderGen
from src.denoiser.diffusion import SEED
from src.tasks.metrics import DraftLoss, NextTokenLoss

log = logging.getLogger(__name__)


def build_eval_metrics(model, metrics_config):
    """Instantiate configured metrics and add raw loss components for joint SEED."""
    metrics = list(hydra.utils.instantiate(metrics_config).values())
    # Only joint SEED backbones produce both next-token and drafting losses.
    # Separate metric classes keep Composer's class-based metric names distinct.
    if isinstance(model, SEED) and isinstance(
        model.backbone, LLMasEncoderDecoderShareKVEncoderGen
    ):
        metrics.extend([NextTokenLoss(), DraftLoss()])
    return metrics


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for training."""
    print_and_save_config(cfg, resolve=True, save_cfg=True)
    reproducibility.seed_all(cfg.seed)

    # Tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    # Model
    model = hydra.utils.instantiate(
        cfg.model,
        _convert_="all",  # required to enable json-serialization when saving checkpoint
    )
    print(model)
    eval_metrics = build_eval_metrics(model, cfg.eval_metrics)
    if getattr(cfg.training, "compile_backbone", False):
        log.info("Compiling model backbone")
        model.backbone = torch.compile(
            model.backbone, dynamic=False, mode="max-autotune-no-cudagraphs"
        )
    model = HuggingFaceModel(
        model,
        tokenizer=tokenizer,
        metrics=list(hydra.utils.instantiate(cfg.metrics).values()),
        eval_metrics=eval_metrics,
    )
    log.info(
        f"Num. parameters: {format_number(count_parameters(model, trainable=False))}"
    )
    log.info(f"Num. trainable parameters: {format_number(count_parameters(model))}")

    # Setup distributed
    if not dist.is_initialized():
        log.info("Initializing dist")
        dist.initialize_dist(timeout=600)
    log.info("All nodes connected")

    # Collator
    train_collator = hydra.utils.instantiate(
        cfg.collator,
        tokenizer=tokenizer,
        rank=dist.get_global_rank(),
        world_size=dist.get_world_size(),
    )
    eval_collator = hydra.utils.instantiate(
        cfg.eval_collator,
        tokenizer=tokenizer,
        rank=dist.get_global_rank(),
        world_size=dist.get_world_size(),
    )

    # Train dataloader
    train_dataset = hydra.utils.instantiate(
        cfg.train_dataset,
        tokenizer=tokenizer,
    )
    train_sampler = dist.get_sampler(train_dataset, shuffle=True, drop_last=True)
    train_dataloader = hydra.utils.instantiate(
        cfg.train_dataloader,
        _convert_="partial",
        dataset=train_dataset,
        collate_fn=train_collator,
        sampler=train_sampler,
    )
    # time.sleep(30)  # Needed for multi-node training

    # Val dataloader (optional)
    if getattr(cfg, "eval_dataset", None) is not None:
        eval_dataset = hydra.utils.instantiate(
            cfg.eval_dataset,
            tokenizer=tokenizer,
        )
        eval_sampler = dist.get_sampler(eval_dataset, shuffle=False, drop_last=False)
        eval_dataloader = hydra.utils.instantiate(
            cfg.eval_dataloader,
            _convert_="partial",
            dataset=eval_dataset,
            collate_fn=eval_collator,
            sampler=eval_sampler,
        )
    else:
        eval_dataset, eval_dataloader = None, None
    # time.sleep(30)  # Needed for multi-node training

    # Optimizer
    optimizer = hydra.utils.instantiate(
        cfg.composer.optimizer,
        _convert_="all",  # required for compatibility with fsdp
        params=model.parameters(),
    )

    # LR Scheduler
    lr_scheduler = hydra.utils.instantiate(cfg.composer.lr_scheduler)

    # Loggers
    if cfg.composer.loggers is not None:
        logger = hydra.utils.instantiate(
            cfg.composer.loggers,
            _recursive_=False,
            # Prevents config->DictConfig in trainer init; breaks WandB config logging
            _convert_="all",
            init_kwargs={"config": OmegaConf.to_container(cfg, resolve=True)},
        )
    else:
        logger = None

    # for debug: set logger to None
    # logger = None

    # Callbacks
    callbacks = hydra.utils.instantiate(cfg.composer.callbacks)

    # Algorithms
    algorithms = hydra.utils.instantiate(cfg.composer.algorithms)

    # Parallelism config (FSDP)
    parallelism_config = None
    if hasattr(cfg.composer, "parallelism") and cfg.composer.parallelism is not None:
        parallelism_config = hydra.utils.instantiate(cfg.composer.parallelism)
        
        # --- FIX START: Surgically remove device_mesh ---
        if hasattr(parallelism_config, 'fsdp'):
            fsdp_conf = parallelism_config.fsdp
            # Check if fsdp is a dictionary/DictConfig and has the forbidden key
            if isinstance(fsdp_conf, (dict, DictConfig)) and 'device_mesh' in fsdp_conf:
                log.info("Manually removing deprecated 'device_mesh' key from FSDP config to prevent crash.")
                if isinstance(fsdp_conf, DictConfig):
                    with open_dict(fsdp_conf):
                        del fsdp_conf['device_mesh']
                else:
                    fsdp_conf.pop('device_mesh', None)
        # --- FIX END ---

        log.info(f"Using parallelism config: {parallelism_config}")

    # Trainer - build kwargs manually to avoid hydra converting ParallelismConfig to dict
    trainer_kwargs = dict(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        algorithms=list(algorithms.values()),
        loggers=logger,
        callbacks=list(callbacks.values()),
    )
    if parallelism_config is not None:
        trainer_kwargs["parallelism_config"] = parallelism_config

    trainer = hydra.utils.instantiate(
        cfg.composer.trainer,
        _convert_="partial",
        **trainer_kwargs,
    )

    trainer.fit()

    if torch_dist.is_initialized():
        torch_dist.destroy_process_group()

    # Clean up `tmp` dir potentially created StreamingDataset
    if hasattr(train_dataset, "remove_tmp_files"):
        train_dataset.remove_tmp_files()
    if eval_dataset is not None and hasattr(eval_dataset, "remove_tmp_files"):
        eval_dataset.remove_tmp_files()


if __name__ == "__main__":
    register_useful_resolvers()
    main()
