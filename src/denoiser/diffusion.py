from functools import partial
from typing import Any, Dict, Literal, Optional, Tuple, Union
import math

import torch
from tqdm.auto import tqdm
from transformers import (
    GenerationConfig,
    LogitsProcessorList,
    PreTrainedTokenizer,
    StoppingCriteriaList,
)
from transformers.cache_utils import Cache, DynamicCache

try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        and_masks,
        create_block_mask,
    )
except ImportError:
    BlockMask, and_masks, create_block_mask = None, None, None


from src.denoiser.base import (
    Denoiser,
    DenoiserConfig,
    DenoiserInput,
    LossAndNllOutput,
)


def create_attn_mask(attn_mask):
    # noinspection PyUnusedLocal
    def padding(b, h, q_idx, kv_idx):
        return attn_mask[b, q_idx] & attn_mask[b, kv_idx]

    return padding


class DiffusionGenerationConfig(GenerationConfig):
    def __init__(
        self,
        num_steps: int = 1000,
        min_t: float = 1e-5,
        block_size: Optional[int] = None,
        first_hitting: bool = False,
        sampling_strategy: Literal["posterior", "predict_then_noise"] = "posterior",
        confidence_based_noising: bool = False,
        confidence_margin_based_noising: bool = False,
        confidence_threshold: float = 1e6,
        use_model_output_cache: bool = True,
        align_inputs_to_blocks: bool = True,
        **kwargs,
    ):
        """Generation config with additional parameters relevant for diffusion model
            sampling.

        Args:
            num_steps (int): Number of diffusion / iterative refinement steps.
                Defaults to 1000.
            min_t (float): Minimum time to use.
                Diffusion models use t=1 for noise and t=0 for signal.
                Setting t=0 exactly can lead to certain numerical instabilities.
                Defaults to 1e-5.
            block_size (int): Block size to use for semi-autoregressive decoding.
                Defaults to None (in which case block_size is set to max_new_tokens).
            first_hitting (bool): Whether to use first hitting sampler.
                When set to true, rather than following the diffusion time and sampling
                from posterior, which can result in no tokens changing between steps,
                e.g., for masked diffusion, we explicitly determine the next time step
                at which a token will be decoded / generated.
                Note: this will negate the `num_steps` parameter, as we will decode one
                token at a time, hence, when True, num_steps = seq_length
                (or block_size, for semi-autoregressive).
                See https://arxiv.org/abs/2409.02908 for details.
                Defaults to False.
            sampling_strategy (str): Method for transitioning between latents.
                Options:
                    - "posterior" - Compute and sample from the posterior
                        q(x_s | x_t, x_theta).
                    - "predict_then_noise" - Sample from the denoising model x_theta,
                        then add back noise to produce x_s.
                        Only implemented for absorbing diffusion.
                Defaults to "posterior".
            confidence_based_noising (bool): When using the "predict_then_noise"
                strategy, whether to add noise to random positions or to those that have
                the lowest probability under x_theta.
                Cannot be used in conjunction with confidence_margin_based_noising.
                Defaults to False.
            confidence_margin_based_noising (bool): When using the "predict_then_noise"
                strategy, whether to add noise to random positions or to those that have
                the lowest probability margins under x_theta, where margin is defined as
                the absolute difference between the top two probabilities at a given
                position.
                See https://arxiv.org/abs/2502.06768 for details.
                Cannot be used in conjunction with confidence_based_noising.
                Defaults to False.
            confidence_threshold (float): Confidence threshold to use for sampling.
                Any tokens that exceed threshold are decoded.
                See https://arxiv.org/abs/2505.22618 for details.
                Defaults to 1e6.
            use_model_output_cache (bool): Whether to re-use model's output, if sequence
                is unchanged, because if xt == xs, we can simply re-use the denoising
                model's outputs and save a function evaluation.
                Relevant if model.backbone is not time/noise-conditioned.
                Defaults to True.
            align_inputs_to_blocks (bool): Whether to align input tokens to block size,
                e.g., for an input of length C and block size S, context will be C // S,
                and generation will begin with a block whose first C % S tokens come
                from the input.
            kwargs: Keyword arguments passed to `GenerationConfig`.
        """
        super().__init__(**kwargs)
        self.num_steps = num_steps
        self.min_t = min_t
        # TODO: assumes we are setting max_new_tokens, which may not be the case!
        self.block_size = block_size if block_size is not None else self.max_new_tokens
        self.first_hitting = first_hitting
        if self.first_hitting:
            # TODO: log.warn that this is being overridden
            self.num_steps = min(num_steps, self.block_size)
        self.sampling_strategy = sampling_strategy
        assert not confidence_based_noising or not confidence_margin_based_noising, (
            "Cannot use both `confidence_based_noising` and"
            " `confidence_margin_based_noising`."
        )
        self.confidence_based_noising = confidence_based_noising
        self.confidence_margin_based_noising = confidence_margin_based_noising
        self.confidence_threshold = confidence_threshold
        self.use_model_output_cache = use_model_output_cache
        self.align_inputs_to_blocks = align_inputs_to_blocks


class D3PMConfig(DenoiserConfig):
    """Configuration class for D3PM models."""

    model_type = "d3pm"
    auto_map = {
        "AutoConfig": "diffusion.D3PMConfig",
        "AutoModel": "diffusion.D3PM",
        "AutoModelForMaskedLM": "diffusion.D3PM",
    }

    def __init__(
        self,
        keep_clean_bos: Optional[bool] = None,  # Whether to enforce un-noised BOS token
        T: int = 1000,
        diffusion_type: Literal["absorbing"] = "absorbing",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.keep_clean_bos = keep_clean_bos
        self.diffusion_type = diffusion_type
        self.T = T


class D3PM(Denoiser):
    """Denoiser class for D3PM models.

    This class implements the Denoiser interface for D3PM models.
    """

    config_class = D3PMConfig

    def __init__(self, config: D3PMConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.T = config.T
        self.diffusion_type = config.diffusion_type
        self._create_static_mask()

    def _create_static_mask(self) -> None:
        static_mask = torch.ones(
            self.config.length, self.config.length, dtype=torch.bool
        )
        self.register_buffer(
            "static_attention_mask",
            static_mask,
        )
        self.skip_params_for_push.append("static_attention_mask")

    def _sample_q_xt(
        self,
        x0: torch.LongTensor,
        alpha_t: torch.FloatTensor,
        context_mask: torch.FloatTensor,
    ) -> torch.LongTensor:
        """Sample from the pre-defined forward / noising process.

        Parameters:
            x0 (Tensor): Signal / data sample;
                can potentially include context tokens.
            alpha_t (Tensor): Amount of signal to retain.
            context_mask (Tensor): Indicator of context tokens (to remain
                unchanged).
        """
        move_indices = torch.rand(*x0.shape, device=x0.device) < (1.0 - alpha_t)
        if self.diffusion_type == "absorbing":
            xt = torch.where(
                (move_indices * (1 - context_mask)).bool(), self.mask_token_id, x0
            )
            if self.config.keep_clean_bos:
                xt[..., 0] = x0[..., 0]
            return xt  # type: ignore
        raise NotImplementedError(
            f"Diffusion type '{self.diffusion_type}' not implemented."
        )

    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
    ):
        # Prepare inputs for D3PM model
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if context_mask is None:
            context_mask = torch.zeros_like(attention_mask)

        if torch.is_floating_point(attention_mask):
            attention_mask = attention_mask.to(torch.int)
            context_mask = context_mask.to(torch.int)

        if t is None:
            t = torch.rand(input_ids.shape[0], device=input_ids.device)
        alpha_t, alpha_t_prime = self.noise_schedule(t)
        while alpha_t.ndim < 2:
            alpha_t = alpha_t[..., None]
            alpha_t_prime = alpha_t_prime[..., None]
        xt = self._sample_q_xt(
            x0=input_ids,
            alpha_t=alpha_t,
            context_mask=context_mask,
        )
        if (
            context_mask is not None
            and context_mask.sum() == 0
            and (attention_mask == 1).all()
        ):
            processed_attention_mask = None
        else:
            processed_attention_mask = (
                self.static_attention_mask[None, ...]
                & attention_mask[:, None, :]
                & attention_mask[..., None]
            )[:, None, ...]  # Make attention mask 4D
            processed_attention_mask = self._preprocess_attention_mask(
                processed_attention_mask, dtype=torch.float
            )
        if self.training and self.config.train_on_context:
            tokens_mask = attention_mask
        else:
            tokens_mask = attention_mask * (1 - context_mask)
        return DenoiserInput(
            xt=xt,
            x0=input_ids,
            attention_mask=processed_attention_mask,
            context_mask=context_mask,
            tokens_mask=tokens_mask,
            t=t,
            alpha_t=alpha_t,
            alpha_t_prime=alpha_t_prime,
        )

    def _prepare_inputs_inference(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        context: Optional[torch.LongTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        cache: Optional[Dict[str, Any]] = None,
        **backbone_kwargs: Any,
    ) -> Tuple[DenoiserInput, Dict[str, Any]]:
        assert input_ids is not None or context is not None, (
            "Must provide either input_ids or context."
        )
        cache = cache if cache is not None else {}
        past_key_values = cache.pop("past_key_values", DynamicCache())
        if context is not None:
            if input_ids is not None:
                if context_mask is None:
                    context_mask = torch.cat(
                        [torch.ones_like(context), torch.zeros_like(input_ids)], dim=-1
                    )
                input_ids = torch.cat([context, input_ids], dim=-1)
            else:
                input_ids = context
                context_mask = torch.ones_like(input_ids)
        if attention_mask is None:
            cache_length = self._get_past_key_values_seq_length(past_key_values)
            full_seq_length = cache_length + input_ids.shape[-1]
            attention_mask = torch.ones(
                (input_ids.shape[0], 1, input_ids.shape[1], full_seq_length),
                device=input_ids.device,
            )  # Make attention mask 4D
            attention_mask = self._preprocess_attention_mask(
                attention_mask, dtype=torch.float
            )
        return DenoiserInput(
            xt=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            context_mask=context_mask,
            backbone_kwargs=backbone_kwargs,
        ), cache

    def _forward(
        self,
        backbone_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs,
    ) -> torch.FloatTensor:
        return torch.log_softmax(backbone_output, dim=-1)  # type: ignore

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        raise NotImplementedError

    def _sample_prior(self, device, batch_size, length):
        """Samples from prior / limiting distribution."""
        if self.diffusion_type == "absorbing":
            return self.mask_token_id * torch.ones(
                (batch_size, length), dtype=torch.int64, device=device
            )
        raise NotImplementedError(
            f"Diffusion type '{self.diffusion_type}' not implemented."
        )

    # noinspection PyUnusedLocal
    def _compute_posterior(
        self,
        x: Union[torch.FloatTensor, torch.LongTensor],
        xt: torch.LongTensor,
        alpha_t: torch.FloatTensor,
        alpha_s: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Computes posterior / approximate posterior q(x_s | x_t, x),
            where x represents clean sequence (as one-hots) or the output of the
            denoising model.

        Args:
            x (Tensor): True (one-hot) / predicted clean signal (B, L, V).
            xt (Tensor): Noised signal at time t (B, L).
            alpha_t (Tensor): Noise schedule parameter at time t (B, 1, 1).
            alpha_s (Tensor): Noise schedule parameter at time s (B, 1, 1).
        """
        if self.diffusion_type == "absorbing":
            q_xs = x * (alpha_s - alpha_t)
            q_xs[..., self.mask_token_id] = 1 - alpha_s[..., 0]
            q_xs /= 1 - alpha_t
            return q_xs  # type: ignore

        raise NotImplementedError(
            f"Diffusion type {self.diffusion_type} not implemented."
        )

    @staticmethod
    def _sample_generation_timesteps(
        generation_config: DiffusionGenerationConfig,
        max_length: Optional[int] = None,
        device: Optional[str] = None,
    ) -> torch.FloatTensor:
        """Sample timesteps for diffusion generation process."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if max_length is None:
            max_length = generation_config.max_new_tokens

        if (
            generation_config.first_hitting
            # TODO: first-hitting does not work with posterior
            and generation_config.sampling_strategy == "posterior"
        ):
            timesteps = torch.FloatTensor([1.0])
            for i in range(max_length, 0, -1):
                u = torch.rand(1)
                next_t = timesteps[-1] * u ** (1 / i)
                timesteps = torch.cat((timesteps, next_t), dim=0)
            return timesteps[1:].to(device)  # type: ignore
        return torch.linspace(  # type: ignore
            1.0,
            generation_config.min_t,
            generation_config.num_steps + 1,
            device=device,
        )[:-1]

    def _generate_unconditional(
        self,
        generation_config: DiffusionGenerationConfig,
        alpha_t: torch.FloatTensor,
        alpha_s: torch.FloatTensor,
        denoiser_inputs: Optional[DenoiserInput] = None,
        model_output_cache: Optional[Dict[str, torch.FloatTensor]] = None,
        cache: Optional[Dict[str, Any]] = None,
        running_generation: Optional[torch.LongTensor] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        **kwargs: Any,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.FloatTensor], Dict[str, Any]]:
        cache = cache if cache is not None else {}
        if model_output_cache is None:  # execute function evaluation
            backbone_output = self._backbone_forward(
                denoiser_inputs,
                fix_cache_length=True,  # Do not let kv cache grow on each forward call
                **cache,
                **kwargs,
            )
            backbone_output = {k: v for k, v in backbone_output.items()}
            logits = backbone_output.pop("logits")
            cache = cache | backbone_output
            log_x_theta = self._forward(logits, denoiser_inputs, **kwargs)
            if logits_processor is not None:
                for token_idx in range(log_x_theta.shape[1]):
                    # TODO: Looping over token positions like this does not allow for
                    #   some processors, e.g. length penalty which could be applied all
                    #   at once to the entire block, to be applied in parallel.
                    log_x_theta[:, token_idx] = logits_processor(
                        input_ids=running_generation,
                        scores=log_x_theta[:, token_idx],  # type: ignore
                    )
                log_x_theta = torch.log_softmax(log_x_theta, dim=-1)  # re-normalize
            x_theta = log_x_theta.exp()
        else:
            x_theta = model_output_cache["x_theta"]
        model_output_cache = {"x_theta": x_theta}
        prob_check_denom = denoiser_inputs.xt.numel()
        if generation_config.sampling_strategy == "posterior":
            q_xs = self._compute_posterior(
                x_theta, denoiser_inputs.xt, alpha_t, alpha_s
            )

            assert abs((q_xs.sum() / prob_check_denom).item() - 1.0) < 1e-6, (
                "Posterior probabilities not summing to 1."
            )
            assert q_xs.isnan().sum().item() == 0, "NaN found in the posterior."
            xs = self._sample_categorical(q_xs, generation_config.do_sample)
            output = torch.where(
                (denoiser_inputs.xt != self.mask_token_id).bool(),  # type: ignore
                denoiser_inputs.xt,
                xs,
            )
        elif generation_config.sampling_strategy == "predict_and_noise":
            assert self.config.diffusion_type == "absorbing", (
                "predict_and_noise decoding strategy only supports absorbing diffusion."
            )
            # assert (
            #     abs((x_theta.sum() / prob_check_denom).item() - 1.0) < 1e-6
            # ), "Denoising output probabilities not summing to 1."
            # assert x_theta.isnan().sum().item() == 0, (
            #     "NaN found in the denoising output."
            # )

            # Predict
            xs = self._sample_categorical(x_theta, generation_config.do_sample)
            xs_probs = x_theta.gather(-1, xs[..., None]).squeeze(dim=-1)
            output = xs.clone()

            # Noise
            num_noise_indices = torch.minimum(
                ((1 - alpha_s) * generation_config.block_size).to(torch.int),
                (denoiser_inputs.xt == self.mask_token_id).sum() - 1,  # type: ignore
            )
            if generation_config.confidence_based_noising:
                conf = x_theta.gather(-1, xs[..., None]).squeeze(-1)
                conf = torch.where(  # already decoded tokens have 'inf' confidence
                    (denoiser_inputs.xt == self.mask_token_id).bool(),  # type: ignore
                    conf,
                    torch.inf,
                )
                noise_indices = conf.argsort(dim=-1)[..., :num_noise_indices]
            elif generation_config.confidence_margin_based_noising:
                top2 = torch.topk(x_theta, k=2, dim=-1).values  # shape: (B, L, 2)
                conf = (top2[..., 0] - top2[..., 1]).abs()
                conf = torch.where(  # already decoded tokens have 'inf' confidence
                    (denoiser_inputs.xt == self.mask_token_id).bool(),  # type: ignore
                    conf,
                    torch.inf,
                )
                noise_indices = conf.argsort(dim=-1)[..., :num_noise_indices]
            else:
                # TODO: implement random noise indices selection
                raise NotImplementedError
            output[..., noise_indices] = self.mask_token_id
            output = torch.where(
                xs_probs >= generation_config.confidence_threshold, xs, output
            )
        else:
            raise NotImplementedError(
                f"Sampling strategy {generation_config.sampling_strategy} not"
                " implemented."
            )
        return output, model_output_cache, cache  # type: ignore

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.LongTensor] = None,
        generation_config: Optional[DiffusionGenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        disable_pbar: bool = False,
        **kwargs: Any,
    ) -> torch.LongTensor:
        # Setup sampling variables
        if generation_config is None:
            assert getattr(self, "generation_config", None) is not None, (
                "Generation config must be provided if not present in the model."
            )
            generation_config = self.generation_config

        # handle unconditional generation when inputs is None
        if inputs is None:
            inputs = torch.ones((batch_size, 1), device=device) * self.bos_token_id # create a starting tensor with bos_tokens

        if max_length is None:
            if hasattr(generation_config, "max_length"):
                max_length = generation_config.max_length
            else:
                max_length = self.max_length
        if max_new_tokens is None:
            if hasattr(generation_config, "max_new_tokens"):
                max_new_tokens = generation_config.max_new_tokens
            else:
                max_new_tokens = max_length - inputs.shape[-1]
        batch_size = batch_size if batch_size is not None else inputs.shape[0]
        assert batch_size == 1, "Batched sampling not supported yet"
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        block_size = generation_config.block_size
        max_blocks = max_new_tokens // block_size

        # create a tensor filled with [MASK]s
        accumulated_samples = self._sample_prior(
            device=device,
            batch_size=batch_size,
            length=max_blocks * block_size,
        )
        accumulated_samples = torch.cat([inputs, accumulated_samples], dim=-1)
        if generation_config.use_cache and inputs.numel() > 0:
            cache = self.update_cache(
                inputs=inputs[:, : block_size * (inputs.shape[-1] // block_size)]
                if generation_config.align_inputs_to_blocks # ensure the prompt length is a multiple of the block size for cleaner processing
                else inputs,
                cache={},
            )
        else:
            cache = None

        # mark the index where the masked region (the generation) begins
        if generation_config.align_inputs_to_blocks:
            inputs_offset = (
                block_size * (inputs.shape[-1] // block_size)
                if inputs.numel() > 0
                else 0
            )
        else:
            inputs_offset = inputs.shape[-1] if inputs.numel() > 0 else 0

        total_NFEs = 0

        # create the noise schedule timesteps (a sequence of t values decreasing from 1.0 to 0.0)
        timesteps = self._sample_generation_timesteps(  # Re-use in every block
            generation_config, max_length=block_size, device=device
        )
        dt = (1 - generation_config.min_t) / len(timesteps)

        # block by block generation
        block_pbar = tqdm(
            range(max_blocks),
            desc="Blocks",
            leave=True,
            disable=disable_pbar,
        )
        for block_id in block_pbar:
            block_NFEs = 0

            # slice the current block to be denoised
            xt = accumulated_samples[
                :,
                inputs_offset + (block_id * block_size) : inputs_offset + ((block_id + 1) * block_size),
            ]

            if self.mask_token_id not in xt:
                continue
            step_pbar = tqdm(
                timesteps,
                desc="T",
                total=timesteps.shape[0],
                leave=False,
                disable=disable_pbar,
            )
            model_output_cache = None
            context = (
                accumulated_samples[:, : (block_id * block_size) + inputs_offset]
                if not generation_config.use_cache
                else None
            )
            # Used for logit processing
            running_generation = accumulated_samples[
                :,
                inputs_offset : inputs_offset + (block_id * block_size),
            ]

            for t in step_pbar:
                if model_output_cache is None:
                    block_NFEs += 1
                    total_NFEs += 1
                # a single denoising step
                # t is 0-dim tensor, reshape to (1, 1, 1) for broadcasting
                alpha_t, _ = self.noise_schedule(t)
                alpha_s, _ = self.noise_schedule(t - dt)
                alpha_t = alpha_t[None, None, None]
                alpha_s = alpha_s[None, None, None]
                denoiser_inputs, cache = self._prepare_inputs_inference(
                    input_ids=xt,
                    context=context,
                    cache=cache if generation_config.use_cache else None,
                )
                xs, model_output_cache, cache = self._generate_unconditional(
                    generation_config=generation_config,
                    alpha_t=alpha_t,
                    alpha_s=alpha_s,
                    denoiser_inputs=denoiser_inputs,
                    model_output_cache=model_output_cache,
                    cache=cache,
                    running_generation=running_generation,  # type: ignore
                    logits_processor=logits_processor,
                    tokenizer=tokenizer,
                    **kwargs,
                )
                block_pbar.set_postfix(
                    NFEs=total_NFEs,
                    block_NFEs=block_NFEs,
                )

                if (
                    not torch.allclose(xs, denoiser_inputs.xt)
                    or not generation_config.use_model_output_cache
                ):
                    model_output_cache = None
                if not generation_config.use_cache:
                    xt[..., -block_size:] = xs[..., -block_size:]
                else:
                    xt = xs
                if (xt == self.mask_token_id).sum().item() == 0 and self.config.diffusion_type == "absorbing":
                    break

            accumulated_samples[
                :,
                inputs_offset + (block_id * block_size) : inputs_offset
                + ((block_id + 1) * block_size),
            ] = xt
            if tokenizer is not None:  # Useful for debugging
                print(tokenizer.batch_decode(accumulated_samples))
            if stopping_criteria is not None:
                is_done = stopping_criteria(
                    input_ids=accumulated_samples[  # type: ignore
                        :,
                        inputs_offset : inputs_offset + ((block_id + 1) * block_size),
                    ],
                    scores=None,  # type: ignore
                )
                if torch.any(is_done):
                    accumulated_samples = accumulated_samples[
                        :,
                        : inputs_offset + ((block_id + 1) * block_size),
                    ]
                    break
            if generation_config.use_cache:
                cache = self.update_cache(
                    inputs=xt,
                    cache=cache,
                )
        return accumulated_samples  # type: ignore


class MDLMConfig(D3PMConfig):
    """Configuration class for MDLM models."""

    model_type = "mdlm"
    auto_map = {
        "AutoConfig": "diffusion.MDLMConfig",
        "AutoModel": "diffusion.MDLM",
        "AutoModelForMaskedLM": "diffusion.MDLM",
    }


class MDLM(D3PM):
    """Denoiser class for MDLM models."""

    config_class = MDLMConfig

    def __init__(self, config: MDLMConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.neg_infinity = -1e12

    def _forward(
        self,
        backbone_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs,
    ) -> torch.FloatTensor:
        # Zero-mask probability
        backbone_output[..., self.mask_token_id] = self.neg_infinity
        log_probs = backbone_output - torch.logsumexp(
            backbone_output, dim=-1, keepdim=True
        )
        # Copy-over unmasked: For the log_probs of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        xt = denoiser_inputs.xt
        unmasked_indices = xt != self.mask_token_id
        log_probs[unmasked_indices] = self.neg_infinity
        log_probs[unmasked_indices, xt[unmasked_indices]] = 0
        return log_probs  # type: ignore

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        log_p_theta = torch.gather(
            input=model_output, dim=-1, index=denoiser_inputs.x0[:, :, None]
        ).squeeze(-1)
        nlls = (
            log_p_theta
            * denoiser_inputs.alpha_t_prime
            / (1 - denoiser_inputs.alpha_t)
            * denoiser_inputs.tokens_mask
        )
        if self.training:
            batch_nll = -(log_p_theta * denoiser_inputs.tokens_mask).sum(dim=-1)
        else:
            batch_nll = nlls.sum(dim=-1)
        count = denoiser_inputs.tokens_mask.sum(dim=-1)
        token_nll = (batch_nll / count).mean()
        return LossAndNllOutput(
            loss=token_nll,  # type: ignore
            nlls=nlls,
            other_loss_terms={
                "masked_tokens": (denoiser_inputs.xt == self.mask_token_id).int()
            },
        )


class BD3LMConfig(MDLMConfig):
    """Configuration class for BD3LM models."""

    model_type = "bd3lm"
    auto_map = {
        "AutoConfig": "diffusion.BD3LMConfig",
        "AutoModel": "diffusion.BD3LM",
        "AutoModelForMaskedLM": "diffusion.BD3LM",
    }

    def __init__(
        self,
        block_size: Optional[int] = None,
        eval_block_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.eval_block_size = (
            eval_block_size if eval_block_size is not None else block_size
        )


class BD3LM(MDLM):
    """Denoiser class for BD3LM models."""

    config_class = BD3LMConfig

    def __init__(self, config: BD3LMConfig, **kwargs):
        super().__init__(config, **kwargs)

    # noinspection PyUnusedLocal
    @staticmethod
    def _block_mask(
        b,
        h,
        q_idx,
        kv_idx,
        block_size: Optional[int] = None,
        seq_length: Optional[int] = None,
    ) -> torch.Tensor:
        del b, h

        # Indicate whether token belongs to xt or x0:
        xt_flag_q = (q_idx >= seq_length).bool()
        xt_flag_kv = (kv_idx >= seq_length).bool()

        # Compute block indices
        block_q = torch.where(
            xt_flag_q, (q_idx - seq_length) // block_size, q_idx // block_size
        )
        block_kv = torch.where(
            xt_flag_kv, (kv_idx - seq_length) // block_size, kv_idx // block_size
        )
        # **1. Offset Block-Causal Mask (M_OBC) **
        offset_block_causal = (block_q > block_kv) & ~xt_flag_kv & xt_flag_q

        # **2. Block Diagonal Mask (M_BD) **
        block_diagonal = (block_q == block_kv) & (xt_flag_q == xt_flag_kv)

        # **3. Block-Causal Mask (M_BC) **
        block_causal = (block_q >= block_kv) & ~xt_flag_kv & ~xt_flag_q

        # **3. Combine Masks **
        return block_diagonal | offset_block_causal | block_causal

    def _create_static_mask(self) -> None:
        if self.config.attn_backend == "sdpa":
            static_mask = self._block_mask(
                b=None,
                h=None,
                q_idx=torch.arange(self.config.length * 2)[:, None],
                kv_idx=torch.arange(self.config.length * 2)[None, :],
                block_size=self.config.block_size
                if self.training
                else self.config.eval_block_size,
                seq_length=self.config.length,
            )
            self.register_buffer(
                "static_attention_mask",
                static_mask,
            )
            self.skip_params_for_push.append("static_attention_mask")
        elif self.config.attn_backend == "flex_attention":
            mask = partial(
                self._block_mask,
                block_size=self.config.block_size
                if self.training
                else self.config.eval_block_size,
                seq_length=self.config.length,
            )
            self.static_attention_mask = create_block_mask(
                mask,
                B=None,
                H=None,
                Q_LEN=self.config.length * 2,
                KV_LEN=self.config.length * 2,
            )

    def _ensure_no_unmasked_blocks(
        self,
        input_ids: torch.LongTensor,
        xt: torch.LongTensor,
        context_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        n_blocks = xt.shape[1] // self.config.block_size
        # If context overlaps w/block, ignore it
        blocks_without_masks = ((xt == self.mask_token_id) + context_mask).reshape(
            -1, n_blocks, self.config.block_size
        ).sum(dim=-1) == 0
        if blocks_without_masks.sum() > 0:
            num_remasks_per_block = torch.randint(
                0,
                self.config.block_size,
                blocks_without_masks.shape,
                device=xt.device,
            )
            rand = torch.rand(xt.shape[0], xt.shape[1], device=xt.device)
            perm_indices = torch.argsort(
                rand.view(xt.shape[0], n_blocks, self.config.block_size),
                stable=True,
                dim=-1,
            )
            remask_indices = perm_indices <= num_remasks_per_block[..., None]
            xt = torch.where(
                remask_indices.view(xt.shape[0], xt.shape[1])
                * blocks_without_masks.repeat_interleave(self.config.block_size, dim=1),
                self.mask_token_id,
                xt,
            )
            if self.config.keep_clean_bos:
                xt[..., 0] = input_ids[..., 0]
        return xt

    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if context_mask is None:
            context_mask = torch.zeros_like(attention_mask)

        if torch.is_floating_point(attention_mask):
            attention_mask = attention_mask.to(torch.int)
            context_mask = context_mask.to(torch.int)

        if t is None:
            t = torch.rand(
                input_ids.shape[0],
                input_ids.shape[1] // self.config.block_size
                if self.training
                else self.config.eval_block_size,
                device=input_ids.device,
            ).repeat_interleave(
                self.config.block_size
                if self.training
                else self.config.eval_block_size,
                dim=-1,
            )
        alpha_t, alpha_t_prime = self.noise_schedule(t)
        while alpha_t.ndim < 2:
            alpha_t = alpha_t[..., None]
            alpha_t_prime = alpha_t_prime[..., None]
        xt = self._sample_q_xt(x0=input_ids, alpha_t=alpha_t, context_mask=context_mask)
        # Ensure each block has at least 1 masked token
        if self.training:
            xt = self._ensure_no_unmasked_blocks(
                input_ids,
                xt,
                context_mask,
            )
        if self.config.attn_backend == "sdpa":
            decoder_attention_mask = (
                self.static_attention_mask[None, ...]
                & attention_mask.repeat(1, 2)[:, None, :]
                & attention_mask.repeat(1, 2)[..., None]
            )[:, None, ...]  # Make attention mask 4D
            decoder_attention_mask = self._preprocess_attention_mask(
                decoder_attention_mask, dtype=torch.float
            )
        elif self.config.attn_backend == "flex_attention":
            if context_mask.any():
                raise NotImplementedError(
                    "flex_attention with context_mask not implemented yet."
                )
            elif attention_mask is not None and (attention_mask != 1).any():
                padding_mask = create_attn_mask(
                    attention_mask.bool().repeat(2, 2).bool()
                )
                dec_masks = [
                    partial(
                        self._block_mask,
                        block_size=self.config.block_size
                        if self.training
                        else self.config.eval_block_size,
                        seq_length=self.config.length,
                    ),
                    padding_mask,
                ]
                decoder_attention_mask = create_block_mask(
                    and_masks(*dec_masks),
                    B=input_ids.shape[0],
                    H=None,
                    Q_LEN=input_ids.shape[1] * 2,
                    KV_LEN=input_ids.shape[1] * 2,
                )
            else:
                decoder_attention_mask = self.static_attention_mask
        else:
            raise ValueError("Unknown backbone backend")
        backbone_input_ids = torch.cat((input_ids, xt), dim=-1)
        position_ids = (
            torch.arange(input_ids.shape[1]).repeat(2).to(input_ids.device)[None, :]
        )
        if self.training and self.config.train_on_context:
            tokens_mask = attention_mask
        else:
            tokens_mask = attention_mask * (1 - context_mask)
        return DenoiserInput(
            xt=backbone_input_ids,  # type: ignore
            x0=input_ids,
            attention_mask=decoder_attention_mask,  # type: ignore
            tokens_mask=tokens_mask,
            t=t,
            alpha_t=alpha_t,
            alpha_t_prime=alpha_t_prime,
            backbone_kwargs={
                "cache_position": position_ids[0],
                "position_ids": position_ids,
            },
        )

    def _prepare_inputs_inference(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        context: Optional[torch.LongTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        cache: Optional[Dict[str, Any]] = None,
        return_updated_cache: bool = False,
        **backbone_kwargs: Dict[str, Any],
    ) -> Tuple[DenoiserInput, Union[Dict[str, Any], None]]:
        device = input_ids.device if input_ids is not None else context.device
        assert input_ids is not None or context is not None, (
            "Must provide either input_ids or context."
        )
        cache = cache if cache is not None else {}
        past_key_values = cache.pop("past_key_values", DynamicCache())
        if context is not None:
            if input_ids is not None:
                input_ids = torch.cat([context, input_ids], dim=-1)
            else:
                input_ids = context
        cache_length = self._get_past_key_values_seq_length(past_key_values)
        full_seq_length = cache_length + input_ids.shape[-1]
        decoder_attention_mask = self.static_attention_mask[
            None,
            None,
            cache_length:full_seq_length,
            :full_seq_length,
        ]  # Make attention mask 4D
        decoder_attention_mask = self._preprocess_attention_mask(
            decoder_attention_mask, dtype=torch.float
        )
        position_ids = torch.arange(cache_length, full_seq_length).to(device)[None, :]
        return DenoiserInput(
            xt=input_ids,
            attention_mask=decoder_attention_mask,
            context_mask=context_mask,
            past_key_values=past_key_values,
            backbone_kwargs={
                "position_ids": position_ids,
            }
            | backbone_kwargs,
        ), cache

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        input_length = denoiser_inputs.xt.shape[1] // 2
        model_output = model_output[:, input_length:, ...]
        return super()._compute_loss(
            model_output=model_output,  # type: ignore
            denoiser_inputs=denoiser_inputs,
            **kwargs,
        )


class E2D2Config(BD3LMConfig):
    """Configuration class for E2D2 models."""

    model_type = "e2d2"
    auto_map = {
        "AutoConfig": "diffusion.E2D2Config",
        "AutoModel": "diffusion.E2D2",
        "AutoModelForMaskedLM": "diffusion.E2D2",
    }

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)


class E2D2(BD3LM):
    """Denoiser class for E2D2 models."""

    config_class = E2D2Config

    def __init__(self, config: E2D2Config, **kwargs):
        super().__init__(config, **kwargs)

    # noinspection PyUnusedLocal
    @staticmethod
    def _encoder_block_mask(
        b,
        h,
        q_idx,
        kv_idx,
        block_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            q_idx (Tensor): Query indices.
            kv_idx (Tensor): Key indices
            b (Optional: int): batch size
            h (Optional: int): number of heads
            block_size (Optional: int): Defines the block structure.

        Returns:
            Encoder block-causal attention mask.
        """

        # Compute block indices
        block_q = q_idx // block_size
        block_kv = kv_idx // block_size

        # ** Block-Causal Mask **
        return block_q >= block_kv

    # noinspection PyUnusedLocal
    @staticmethod
    def _decoder_block_mask(
        b,
        h,
        q_idx,
        kv_idx,
        block_size: Optional[int] = None,
        seq_length: Optional[int] = None,
    ) -> torch.Tensor:
        # Indicate whether token belongs to xt or x0:
        xt_flag_kv = (kv_idx >= seq_length).bool()

        # Compute block indices
        block_q = q_idx // block_size
        block_kv = torch.where(
            xt_flag_kv, (kv_idx - seq_length) // block_size, kv_idx // block_size
        )
        # **1. Offset Block-Causal Mask (M_OBC) **
        offset_block_causal = (block_q > block_kv) & ~xt_flag_kv

        # **2. Block Diagonal Mask (M_BD) **
        block_diagonal = (block_q == block_kv) & xt_flag_kv

        # **3. Combine Masks **
        return block_diagonal | offset_block_causal

    def _create_static_mask(self) -> None:
        if self.config.attn_backend == "flex_attention":
            enc_mask = partial(
                self._encoder_block_mask,
                block_size=self.config.block_size
                if self.training
                else self.config.eval_block_size,
            )
            encoder_attention_mask = create_block_mask(
                enc_mask,
                B=None,
                H=None,
                Q_LEN=self.config.length,
                KV_LEN=self.config.length,
            )
            dec_mask = partial(
                self._decoder_block_mask,
                block_size=self.config.block_size
                if self.training
                else self.config.eval_block_size,
                seq_length=self.config.length,
            )
            decoder_attention_mask = create_block_mask(
                dec_mask,
                B=None,
                H=None,
                Q_LEN=self.config.length,
                KV_LEN=self.config.length * 2,
            )
            self.encoder_static_attention_mask = encoder_attention_mask
            self.static_attention_mask = decoder_attention_mask
        else:
            if self.config.block_size != "random":
                encoder_static_mask = self._encoder_block_mask(
                    b=None,  # type: ignore
                    h=None,  # type: ignore
                    q_idx=torch.arange(self.config.length)[:, None],
                    kv_idx=torch.arange(self.config.length)[None, :],
                    block_size=self.config.block_size
                    if self.training
                    else self.config.eval_block_size,
                )
                decoder_static_mask = self._decoder_block_mask(
                    b=None,
                    h=None,
                    q_idx=torch.arange(self.config.length)[:, None],
                    kv_idx=torch.arange(self.config.length * 2)[None, :],
                    block_size=self.config.block_size
                    if self.training
                    else self.config.eval_block_size,
                    seq_length=self.config.length,
                )
                self.register_buffer(
                    "encoder_static_attention_mask",
                    encoder_static_mask,
                )
                self.register_buffer(
                    "static_attention_mask",
                    decoder_static_mask,
                )
            else:
                # ---- Modification (11.20): create two static masks for block_size=1 and block_size=4 ----
                encoder_static_mask_1 = self._encoder_block_mask(
                    b=None,  # type: ignore
                    h=None,  # type: ignore
                    q_idx=torch.arange(self.config.length)[:, None],
                    kv_idx=torch.arange(self.config.length)[None, :],
                    # block_size=self.config.block_size
                    block_size=1
                    if self.training
                    else self.config.eval_block_size,
                )
                decoder_static_mask_1 = self._decoder_block_mask(
                    b=None,
                    h=None,
                    q_idx=torch.arange(self.config.length)[:, None],
                    kv_idx=torch.arange(self.config.length * 2)[None, :],
                    # block_size=self.config.block_size
                    block_size=1
                    if self.training
                    else self.config.eval_block_size,
                    seq_length=self.config.length,
                )
                self.register_buffer(
                    "encoder_static_attention_mask_1",
                    encoder_static_mask_1,
                )
                self.register_buffer(
                    "static_attention_mask_1",
                    decoder_static_mask_1,
                )
                self.skip_params_for_push.append("encoder_static_attention_mask_1")
                self.skip_params_for_push.append("static_attention_mask_1")
                encoder_static_mask_4 = self._encoder_block_mask(
                    b=None,  # type: ignore
                    h=None,  # type: ignore
                    q_idx=torch.arange(self.config.length)[:, None],
                    kv_idx=torch.arange(self.config.length)[None, :],
                    # block_size=self.config.block_size
                    block_size=4
                    if self.training
                    else self.config.eval_block_size,
                )
                decoder_static_mask_4 = self._decoder_block_mask(
                    b=None,
                    h=None,
                    q_idx=torch.arange(self.config.length)[:, None],
                    kv_idx=torch.arange(self.config.length * 2)[None, :],
                    # block_size=self.config.block_size
                    block_size=4
                    if self.training
                    else self.config.eval_block_size,
                    seq_length=self.config.length,
                )
                self.register_buffer(
                    "encoder_static_attention_mask_4",
                    encoder_static_mask_4,
                )
                self.register_buffer(
                    "static_attention_mask_4",
                    decoder_static_mask_4,
                )
                self.skip_params_for_push.append("encoder_static_attention_mask_4")
                self.skip_params_for_push.append("static_attention_mask_4")
            # ---- Modification ends ----

    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if context_mask is None:
            context_mask = torch.zeros_like(attention_mask)

        if torch.is_floating_point(attention_mask):
            attention_mask = attention_mask.to(torch.int)
            context_mask = context_mask.to(torch.int)

        if t is None:
            # the first part torch.rand(...) generates one random t per block
            # the repeat_interleave expands it to every token in the block
            # (ensuring that each token in the block has the same t)
            t = torch.rand(
                input_ids.shape[0],                                # batch size
                input_ids.shape[1] // self.config.block_size       # number of blocks
                if self.training
                else self.config.eval_block_size,
                device=input_ids.device,
            ).repeat_interleave(
                self.config.block_size                             # repeat = block size
                if self.training
                else self.config.eval_block_size,
                dim=-1,
            )

        # alpha_t: the probability that a token from the original sequence survives the noising process at time t
        # alpha_t_prime: the speed at which noise is being added
        alpha_t, alpha_t_prime = self.noise_schedule(t)

        # enable the comparison `move_indices = torch.rand(*x0.shape, ...) < (1.0 - alpha_t)` in _sample_q_xt
        while alpha_t.ndim < 2:
            alpha_t = alpha_t[..., None]  # equivalent to torch.unsqueeze(alpha_t, dim=-1)
            alpha_t_prime = alpha_t_prime[..., None] # equivalent to torch.unsqueeze(alpha_t_prime, dim=-1)

        # perform the noising process to get xt from x0
        # context_mask ensures context tokens are not noised
        xt = self._sample_q_xt(x0=input_ids, alpha_t=alpha_t, context_mask=context_mask)

        # Ensure each block has at least 1 masked token
        if self.training:
            xt = self._ensure_no_unmasked_blocks(
                input_ids,
                xt,
                context_mask,
            )

        # handle attention masks
        if self.config.attn_backend == "sdpa":
            decoder_attention_mask = (
                # static_attention_mask (L, 2L) is the offset block-causal mask + block-diagonal mask for the decoder
                self.static_attention_mask[None, ...]
                # attention_mask zeros out rows and cols corresponding to padding tokens
                & attention_mask.repeat(1, 2)[:, None, :] # repeats padding mask to cover 2L
                & attention_mask[..., None]
            )[:, None, ...]  # Make attention mask 4D
            encoder_attention_mask = (
                (
                    # encoder_static_attention_mask (L, L) is just the block-causal encoder mask
                    self.encoder_static_attention_mask[None, ...]
                    # context_mask (L, L) sets the positions corresponding to context (prompt) tokens to 1
                    | context_mask[:, None, :]
                )
                # attention_mask zeros out rows and cols corresponding to padding tokens
                & attention_mask[:, None, :]
                & attention_mask[..., None]
            )[:, None, ...]  # Make attention mask 4D

            # convert binary attention masks (composed of 1s and 0s)
            # into additive attention masks (composed of 0.0 and a very large negative number)
            encoder_attention_mask = self._preprocess_attention_mask(
                encoder_attention_mask, dtype=torch.float
            )
            decoder_attention_mask = self._preprocess_attention_mask(
                decoder_attention_mask, dtype=torch.float
            )
        elif self.config.attn_backend == "flex_attention":
            # TODO enable bidirectional attention on context for seq2seq tasks
            if context_mask.any():
                raise NotImplementedError(
                    "flex_attention with context_mask not implemented yet."
                )
            elif attention_mask is not None and (attention_mask != 1).any():
                padding_mask = create_attn_mask(attention_mask.bool())
                dec_padding_mask = create_attn_mask(attention_mask.repeat(1, 2).bool())
                enc_masks = [
                    partial(
                        self._encoder_block_mask,
                        block_size=self.config.block_size
                        if self.training
                        else self.config.eval_block_size,
                    ),
                    padding_mask,
                ]
                encoder_attention_mask = create_block_mask(
                    and_masks(*enc_masks),
                    B=input_ids.shape[0],
                    H=None,
                    Q_LEN=input_ids.shape[1],
                    KV_LEN=input_ids.shape[1],
                )
                dec_masks = [
                    partial(
                        self._decoder_block_mask,
                        block_size=self.config.block_size
                        if self.training
                        else self.config.eval_block_size,
                        seq_length=input_ids.shape[1],
                    ),
                    dec_padding_mask,
                ]
                decoder_attention_mask = create_block_mask(
                    and_masks(*dec_masks),
                    B=input_ids.shape[0],
                    H=None,
                    Q_LEN=input_ids.shape[1],
                    KV_LEN=input_ids.shape[1] * 2,
                )
            else:
                encoder_attention_mask = self.encoder_static_attention_mask
                decoder_attention_mask = self.static_attention_mask
        else:
            raise ValueError("Unknown backbone backend")
        
        # create a tensor [[0, 1, 2, ..., L-1]]
        position_ids = torch.arange(input_ids.shape[1]).to(input_ids.device)[None, :]

        if self.training and self.config.train_on_context:
            tokens_mask = attention_mask
        else:
            tokens_mask = attention_mask * (1 - context_mask) # only compute loss on non-context tokens
        return DenoiserInput(
            xt=xt,
            x0=input_ids,
            attention_mask=decoder_attention_mask,
            tokens_mask=tokens_mask,
            t=t,
            alpha_t=alpha_t,
            alpha_t_prime=alpha_t_prime,
            backbone_kwargs={
                "encoder_input_ids": input_ids, # encoder uses clean sequence as input
                "encoder_attention_mask": encoder_attention_mask,
                "encoder_position_ids": position_ids,
                "encoder_cache_position": position_ids[0],
            },
        )

    def _prepare_inputs_inference(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        context: Optional[torch.LongTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        cache: Optional[Dict[str, Any]] = None,
        return_updated_cache: bool = False,
        **backbone_kwargs: Dict[str, Any],
    ) -> Tuple[DenoiserInput, Union[Dict[str, Any], None]]:
        device = input_ids.device if input_ids is not None else context.device
        batch_size = input_ids.shape[0] if input_ids is not None else context.shape[0]
        assert input_ids is not None or context is not None, (
            "Must provide either input_ids or context."
        )
        # Indicates this is a cache update step for the encoder
        if return_updated_cache:
            # re-label the clean tokens as context (because they'll serve as context for the next block's generation)
            context = input_ids
            input_ids = None
        # otherwise we're running a denoising step of the decoder, input_ids remains the noisy block xt

        position_ids, encoder_position_ids = None, None

        if cache is not None:
            past_key_values = cache.pop("past_key_values", DynamicCache()) # decoder's cache
            encoder_past_key_values = cache.pop(
                "encoder_past_key_values", DynamicCache()
            )
            encoder_last_hidden_state = cache.pop("encoder_last_hidden_state", None)
            # decoder's run
            if input_ids is not None:
                # check how many tokens the decoder has seen before
                cache_length = self._get_past_key_values_seq_length(past_key_values)
                if encoder_last_hidden_state is not None:
                    full_seq_length = (
                        cache_length
                        + encoder_last_hidden_state.shape[1]  # type: ignore
                        + input_ids.shape[-1]
                    )
                else:
                    full_seq_length = cache_length + input_ids.shape[-1]
                encoder_attention_mask = None
                position_ids = torch.arange(
                    cache_length, full_seq_length, device=device
                )[None, :]
            # encoder's run
            else:
                encoder_cache_length = self._get_past_key_values_seq_length(
                    encoder_past_key_values
                    if encoder_past_key_values is not None and len(encoder_past_key_values) > 0
                    else past_key_values
                )
                encoder_full_seq_length = encoder_cache_length + context.shape[-1]
                encoder_attention_mask = torch.ones(
                    (
                        1,
                        1,
                        encoder_full_seq_length - encoder_cache_length,
                        encoder_full_seq_length,
                    ),
                    device=context.device,
                )
                encoder_position_ids = torch.arange(
                    encoder_cache_length, encoder_full_seq_length
                ).to(device)[None, :]
                encoder_attention_mask = self._preprocess_attention_mask(
                    encoder_attention_mask, dtype=self.backbone.encoder.dtype
                )
                full_seq_length = -1  # Not used
        else:  # Not using kv-cache
            past_key_values = None
            encoder_past_key_values, encoder_last_hidden_state = None, None
            if context is not None:
                context_len = context.shape[1]
                encoder_attention_mask = torch.ones(
                    (1, 1, context_len, context_len), device=context.device
                )
                encoder_attention_mask = self._preprocess_attention_mask(
                    encoder_attention_mask, dtype=self.backbone.encoder.dtype
                )
                encoder_position_ids = torch.arange(context_len).to(device)[None, :]
            else:
                context_len = 0
                encoder_attention_mask = None
            if input_ids is not None:
                full_seq_length = context_len + input_ids.shape[1]
            else:
                full_seq_length = context_len
            position_ids = torch.arange(context_len, full_seq_length).to(device)[
                None, :
            ]
        
        # decoder's run
        if input_ids is not None:
            # can attend to all the tokens
            decoder_attention_mask = torch.ones(
                (batch_size, 1, input_ids.shape[1], full_seq_length),
                device=device,
            )  # Make attention mask 4D
            decoder_attention_mask = self._preprocess_attention_mask(
                decoder_attention_mask, dtype=self.backbone.encoder.dtype
            )
        # encoder's run
        else:
            decoder_attention_mask = None

        # for encoder:
        # xt is None
        # attention_mask is None
        # context_mask is still just passed
        # past_key_values is the decoder's cache
        return DenoiserInput(
            xt=input_ids,
            attention_mask=decoder_attention_mask,
            context_mask=context_mask,
            past_key_values=past_key_values,
            backbone_kwargs={
                "position_ids": position_ids,
                "encoder_input_ids": context,
                "encoder_position_ids": encoder_position_ids,
                "encoder_attention_mask": encoder_attention_mask,
                "encoder_past_key_values": encoder_past_key_values,
                "encoder_last_hidden_state": encoder_last_hidden_state,
            }
            | backbone_kwargs,
        ), cache  # TODO: potentially returning cache None, violates return type

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        # Use MDLM `_compute_loss`, since BD3LM method splits model_output
        return super(BD3LM, self)._compute_loss(
            model_output=model_output,
            denoiser_inputs=denoiser_inputs,
            **kwargs,
        )

class SEEDConfig(E2D2Config):
    """Configuration class for SEED models (autoregressive variant)."""
    
    model_type = "seed"
    auto_map = {
        "AutoConfig": "diffusion.SEEDConfig",
        "AutoModel": "diffusion.SEED",
        "AutoModelForCausalLM": "diffusion.SEED",
    }

    def __init__(
        self,
        decoder_loss_lambda: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.decoder_loss_lambda = decoder_loss_lambda

class SEED(E2D2):
    """Autoregressive variant of E2D2 using block causal attention."""
    
    config_class = SEEDConfig

    def __init__(self, config: SEEDConfig, **kwargs):
        super().__init__(config, **kwargs)

    @staticmethod
    def _speculative_sample(draft_tokens, draft_probs, target_probs):
        """Sample an accepted prefix and one residual/bonus token.

        q has shape (batch, draft_length, vocab); p includes one extra position.
        Implements Algorithm 1 of Leviathan et al. (2023), using fp32 probabilities.
        """
        q = draft_probs.float()
        p = target_probs.float()
        q_token = q.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
        p_token = p[:, :-1].gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
        # Equivalent to U < min(1, p(x)/q(x)), without division by tiny q(x).
        accepted = torch.rand_like(q_token) * q_token < p_token
        lengths = accepted.long().cumprod(dim=1).sum(dim=1)
        rows = torch.arange(q.shape[0], device=q.device)
        correction_probs = p[rows, lengths].clone()
        rejected = lengths < draft_tokens.shape[1]
        correction_probs[rejected] = (
            correction_probs[rejected] - q[rows[rejected], lengths[rejected]]
        ).clamp_min(0)
        # multinomial normalizes the positive residual; when every proposal is
        # accepted, this is already the target distribution for the bonus token.
        correction = torch.multinomial(correction_probs, 1).squeeze(-1)
        return lengths, correction

    @torch.no_grad()
    def _generate_stochastic(
        self,
        inputs,
        generation_config,
        logits_processor,
        stopping_criteria,
        max_length,
        max_new_tokens,
        batch_size,
        device,
        tokenizer,
        disable_pbar,
        conf_seg,
        track_acc_rate,
        tree_attn,
        **kwargs,
    ):
        """Exact speculative sampling, or greedy verification for a frozen target.

        Tree alternatives may supply the residual token and its target continuation.
        As in greedy decoding, a reused tree token counts as one accepted draft;
        the following target token (or an ordinary correction/bonus) does not.
        """
        import time

        do_sample = bool(generation_config.do_sample)
        temperature = (
            float(getattr(generation_config, "temperature", 1.0))
            if do_sample
            else 1.0
        )
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(
                "Stochastic SEED decoding requires a finite temperature > 0."
            )
        if not generation_config.use_cache:
            raise ValueError(
                "Exact SEED speculative decoding requires use_cache=True."
            )
        if inputs is None:
            inputs = torch.full(
                (batch_size or 1, 1), self.bos_token_id, device=device, dtype=torch.long
            )
        if inputs.shape[0] != 1 or batch_size not in (None, 1):
            raise ValueError("Batched SEED speculative decoding is not supported.")
        if inputs.shape[1] == 0:
            raise ValueError("SEED speculative decoding requires a nonempty prompt.")
        device = inputs.device
        if max_new_tokens is None:
            max_new_tokens = getattr(generation_config, "max_new_tokens", None)
        if max_new_tokens is None:
            max_length = max_length or generation_config.max_length
            max_new_tokens = max_length - inputs.shape[1]
        block_size = generation_config.block_size
        if block_size < 1 or max_new_tokens < 0:
            raise ValueError(
                "block_size must be positive and max_new_tokens nonnegative."
            )

        # Stochastic generation always conditions on the entire supplied prompt.
        inputs_offset = inputs.shape[1]
        limit = inputs_offset + max_new_tokens
        samples = torch.cat(
            [
                inputs,
                torch.full(
                    (1, max_new_tokens),
                    self.mask_token_id,
                    device=device,
                    dtype=torch.long,
                ),
            ],
            dim=-1,
        )
        current_idx = inputs_offset
        draft_lengths, accepted_lengths = [], []
        attempts, accepts = [0] * block_size, [0] * block_size
        total_drafted = total_accepted = 0
        draft_events = []
        drafting_time = 0.0
        use_cuda_events = device.type == "cuda"
        if use_cuda_events:
            overall_start = torch.cuda.Event(enable_timing=True)
            overall_end = torch.cuda.Event(enable_timing=True)
            overall_start.record()
        else:
            overall_start = time.perf_counter()

        cache = {"past_key_values": DynamicCache()}
        if inputs_offset > 1 and max_new_tokens > 0:
            cache = self.update_cache(
                inputs=inputs[:, :-1],
                cache=cache,
                enforce_causal_mask=True,
            )

        def processed_scores(logits, prefix):
            if logits_processor is not None:
                logits = logits_processor(input_ids=prefix, scores=logits)
            return logits

        def probabilities(logits):
            return (logits.float() / temperature).softmax(dim=-1)

        def should_stop(end):
            if end >= limit:
                return True
            return stopping_criteria is not None and bool(
                torch.any(
                    stopping_criteria(
                        input_ids=samples[:, inputs_offset:end],
                        scores=None,
                    )
                )
            )

        acc_rate_history = []
        ema_acc_rate = 0.8
        current_draft_len = min(block_size, 10)
        pbar = tqdm(
            total=max_new_tokens, desc="Tokens", leave=True, disable=disable_pbar
        )
        while current_idx < limit:
            start_idx = current_idx
            max_draft_len = (
                current_draft_len
                if track_acc_rate
                else (100 if conf_seg else block_size)
            )
            max_draft_len = min(max_draft_len, limit - start_idx)
            if use_cuda_events:
                draft_start = torch.cuda.Event(enable_timing=True)
                draft_end = torch.cuda.Event(enable_timing=True)
                draft_start.record()
            else:
                draft_start = time.perf_counter()

            draft_probs, candidates = [], []
            for step in range(max_draft_len):
                cur_pos = start_idx + step
                denoiser_inputs, _ = self._prepare_inputs_inference(
                    input_ids=samples[:, cur_pos - 1 : cur_pos],
                    cache=cache,
                )
                output = dict(
                    self._backbone_forward(
                        denoiser_inputs,
                        fix_cache_length=True,
                        truncate_cache=False,
                        **cache,
                        **kwargs,
                    )
                )
                logits = output.pop("logits")
                cache = cache | output
                logits = logits[:, -1] if logits.ndim == 3 else logits
                logits = processed_scores(logits, samples[:, :cur_pos])
                q = probabilities(logits)
                token = (
                    torch.multinomial(q, 1)
                    if do_sample
                    else logits.argmax(dim=-1, keepdim=True)
                )
                samples[:, cur_pos : cur_pos + 1] = token
                draft_probs.append(q)
                confidence = q.max().item()
                if tree_attn:
                    k = (
                        1
                        if confidence > 0.95
                        else (
                            3 if confidence > 0.8 else (5 if confidence > 0.5 else 10)
                        )
                    )
                    for token_id in (
                        q.topk(min(k, q.shape[-1]), dim=-1).indices[0].tolist()
                    ):
                        if token_id != token.item():
                            candidates.append((step, token_id))
                if should_stop(cur_pos + 1) or (conf_seg and confidence < 0.7):
                    break

            draft_len = len(draft_probs)
            total_drafted += draft_len
            # Remove decoder-only speculative KV entries before target scoring.
            kv = cache["past_key_values"]
            for layer in range(len(kv)):
                if kv.key_cache[layer].numel() == 0:
                    continue
                kv.key_cache[layer] = kv.key_cache[layer][..., : start_idx - 1, :]
                kv.value_cache[layer] = kv.value_cache[layer][..., : start_idx - 1, :]
            if use_cuda_events:
                draft_end.record()
                draft_events.append((draft_start, draft_end))
            else:
                drafting_time += time.perf_counter() - draft_start

            main_inputs = samples[:, start_idx - 1 : start_idx + draft_len]
            main_len = main_inputs.shape[1]
            encoder_inputs = main_inputs
            overrides = {}
            if candidates:
                encoder_inputs = torch.cat(
                    [
                        main_inputs,
                        torch.tensor(
                            [[token for _, token in candidates]], device=device
                        ),
                    ],
                    dim=1,
                )
                past_len = start_idx - 1
                positions = list(range(past_len, past_len + main_len))
                positions.extend(start_idx + step for step, _ in candidates)
                overrides["encoder_position_ids"] = torch.tensor(
                    [positions], device=device
                )
                total_len = encoder_inputs.shape[1]
                mask = torch.full(
                    (1, 1, total_len, past_len + total_len),
                    float("-inf"),
                    dtype=self.backbone.encoder.dtype,
                    device=device,
                )
                mask[..., :past_len] = 0
                mask[:, :, :main_len, past_len : past_len + main_len] = torch.triu(
                    torch.full((main_len, main_len), float("-inf"), device=device),
                    diagonal=1,
                )
                for i, (step, _) in enumerate(candidates):
                    row = main_len + i
                    mask[:, :, row, past_len : past_len + step + 1] = 0
                    mask[:, :, row, past_len + row] = 0
                overrides["encoder_attention_mask"] = mask

            verify_kwargs = dict(kwargs)
            verify_kwargs.update(overrides)
            verify_kwargs["enforce_causal_mask"] = not candidates
            verify_kwargs["preserve_encoder_attention_mask"] = bool(candidates)
            temp_cache = cache.copy()
            verifier_inputs, _ = self._prepare_inputs_inference(
                input_ids=encoder_inputs,
                cache=temp_cache,
                return_updated_cache=True,
                **verify_kwargs,
            )
            output = self._backbone_forward(
                verifier_inputs,
                return_updated_cache=True,
                return_last_hidden_state=True,
                **temp_cache,
            )
            target_logits = self.backbone.encoder.lm_head(output["last_hidden_state"])
            target_scores = torch.stack(
                [
                    processed_scores(target_logits[:, i], samples[:, : start_idx + i])
                    for i in range(main_len)
                ],
                dim=1,
            )
            if do_sample:
                lengths, correction = self._speculative_sample(
                    main_inputs[:, 1:],
                    torch.stack(draft_probs, dim=1),
                    probabilities(target_scores),
                )
            else:
                target_tokens = target_scores.argmax(dim=-1)
                lengths = (
                    (main_inputs[:, 1:] == target_tokens[:, :-1])
                    .long()
                    .cumprod(dim=-1)
                    .sum(dim=-1)
                )
                correction = target_tokens.gather(1, lengths[:, None]).squeeze(1)
            accepted = lengths.item()
            final_accepted = accepted
            current_idx = start_idx + accepted
            finished = accepted > 0 and should_stop(current_idx)
            branch_index = None
            if not finished:
                samples[:, current_idx] = correction
                current_idx += 1
                if accepted < draft_len:
                    branch_index = next(
                        (
                            i
                            for i, (step, token) in enumerate(candidates)
                            if step == accepted and token == correction.item()
                        ),
                        None,
                    )
                if branch_index is not None:
                    # Same tree accounting as greedy: the reused branch token
                    # is accepted; its target continuation is a bonus.
                    final_accepted += 1
                finished = should_stop(current_idx)
                if branch_index is not None and not finished:
                    branch_scores = processed_scores(
                        target_logits[:, main_len + branch_index],
                        samples[:, :current_idx],
                    )
                    samples[:, current_idx : current_idx + 1] = (
                        torch.multinomial(probabilities(branch_scores), 1)
                        if do_sample
                        else branch_scores.argmax(dim=-1, keepdim=True)
                    )
                    current_idx += 1
                    finished = should_stop(current_idx)

            # Include terminal verification rounds, and exclude correction/bonus
            # tokens from accepted lengths except for the reused tree token above.
            total_accepted += final_accepted
            accepted_lengths.append(final_accepted)
            draft_lengths.append(draft_len)
            while len(attempts) < draft_len:
                attempts.append(0)
                accepts.append(0)
            for pos in range(draft_len):
                attempts[pos] += 1
                accepts[pos] += int(pos < final_accepted)

            # Keep target KV entries for the accepted prefix and any reused tree
            # branch, excluding the final target continuation and all siblings.
            kv = output["past_key_values"]
            indices = list(range(start_idx + accepted))
            if branch_index is not None:
                indices.append(start_idx - 1 + main_len + branch_index)
            indices = torch.tensor(indices, device=device, dtype=torch.long)
            for layer in range(len(kv)):
                kv.key_cache[layer] = kv.key_cache[layer].index_select(-2, indices)
                kv.value_cache[layer] = kv.value_cache[layer].index_select(-2, indices)
            cache["past_key_values"] = kv

            if track_acc_rate:
                acc_rate_history.append((accepted, draft_len))
                window_accepted = window_drafted = 0
                for hist_accepted, hist_drafted in reversed(acc_rate_history):
                    window_accepted += hist_accepted
                    window_drafted += hist_drafted
                    if window_drafted >= 20:
                        break
                ema_acc_rate = (
                    0.3 * window_accepted / window_drafted + 0.7 * ema_acc_rate
                )
                if ema_acc_rate > 0.8:
                    current_draft_len = min(current_draft_len + 1, 10)
                elif ema_acc_rate < 0.7:
                    current_draft_len = max(current_draft_len - 1, 1)
            pbar.update(current_idx - start_idx)
            if finished:
                break
        pbar.close()
        self._last_draft_position_acceptance = {
            "attempt_counts": attempts,
            "accept_counts": accepts,
            "draft_lengths": draft_lengths,
            "acceptance_rates": [
                acc / att if att else 0.0 for acc, att in zip(accepts, attempts)
            ],
        }
        if use_cuda_events:
            overall_end.record()
            torch.cuda.synchronize()
            total_time = overall_start.elapsed_time(overall_end) / 1000.0
            drafting_time = (
                sum(start.elapsed_time(end) for start, end in draft_events) / 1000.0
            )
        else:
            total_time = time.perf_counter() - overall_start
        if tokenizer is not None:
            output_text = tokenizer.decode(samples[0, inputs_offset:current_idx])
            print(f"[{'STOCHASTIC' if do_sample else 'GREEDY'} OUTPUT] {output_text}")
        return (
            samples[:, :current_idx],
            (total_drafted, total_accepted),
            (accepted_lengths, len(accepted_lengths)),
            (drafting_time, total_time),
        )

    @staticmethod
    def _encoder_block_mask(
        b,
        h,
        q_idx,
        kv_idx,
        block_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Encoder uses block causal masking (same as E2D2)."""
        # Keep the same encoder masking as E2D2
        # block_q = q_idx // block_size
        # block_kv = kv_idx // block_size
        # return block_q >= block_kv
        return q_idx >= kv_idx

    # @staticmethod  <-- Removed staticmethod
    def _decoder_block_mask(
        self,
        b,
        h,
        q_idx,    # tensor [1, L]
        kv_idx,   # tensor [1, 2L]
        block_size: Optional[int] = None,
        seq_length: Optional[int] = None, # L
    ) -> torch.Tensor:
        """Decoder uses block causal attention for AR generation."""

        # Indicate whether token belongs to xt or x0:
        xt_flag_kv = (kv_idx >= seq_length).bool()

        # Compute block indices
        block_q = q_idx // block_size
        block_kv = torch.where(
            xt_flag_kv, (kv_idx - seq_length) // block_size, kv_idx // block_size
        )
        # **1. Offset Block-Causal Mask **
        offset_block_causal = (block_q > block_kv) & ~xt_flag_kv

        # **2. Causal self-attention within decoder blocks**
        # Block causal: causal within current block
        same_block = (block_q == block_kv) & xt_flag_kv
        within_block_causal = same_block & (q_idx >= (kv_idx - seq_length))

        # **3. Combine Masks **
        return offset_block_causal | within_block_causal

    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
    ):
        """Prepare inputs for autoregressive training (no diffusion noise)."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if context_mask is None:
            context_mask = torch.zeros_like(attention_mask)

        if torch.is_floating_point(attention_mask):
            attention_mask = attention_mask.to(torch.int)
            context_mask = context_mask.to(torch.int)

        # For AR: no noise sampling, just use the input as is
        xt = input_ids  # No diffusion noise for autoregressive
        
        # Create attention masks
        if self.config.attn_backend == "sdpa":
            if self.config.block_size == "random":
                # ---- Modification (11.20): select static masks based on randomly sampled block_size ----
                if torch.rand(1).item() < 0.5:
                    decoder_attention_mask = (
                        self.static_attention_mask_1[None, ...]
                        & attention_mask.repeat(1, 2)[:, None, :]
                        & attention_mask[..., None]
                    )[:, None, ...]  # Make attention mask 4D
                    encoder_attention_mask = (
                        (
                            self.encoder_static_attention_mask_1[None, ...]
                            | context_mask[:, None, :]
                        )
                        & attention_mask[:, None, :]
                        & attention_mask[..., None]
                    )[:, None, ...]  # Make attention mask 4D
                    encoder_attention_mask = self._preprocess_attention_mask(
                        encoder_attention_mask, dtype=torch.float
                    )
                    decoder_attention_mask = self._preprocess_attention_mask(
                        decoder_attention_mask, dtype=torch.float
                    )
                else:
                    decoder_attention_mask = (
                        self.static_attention_mask_4[None, ...]
                        & attention_mask.repeat(1, 2)[:, None, :]
                        & attention_mask[..., None]
                    )[:, None, ...]  # Make attention mask 4D
                    encoder_attention_mask = (
                        (
                            self.encoder_static_attention_mask_4[None, ...]
                            | context_mask[:, None, :]
                        )
                        & attention_mask[:, None, :]
                        & attention_mask[..., None]
                    )[:, None, ...]  # Make attention mask 4D
                    encoder_attention_mask = self._preprocess_attention_mask(
                        encoder_attention_mask, dtype=torch.float
                    )
                    decoder_attention_mask = self._preprocess_attention_mask(
                        decoder_attention_mask, dtype=torch.float
                    )
                # ---- Modification ends ----
            else:
                decoder_attention_mask = (
                    self.static_attention_mask[None, ...]
                    & attention_mask.repeat(1, 2)[:, None, :]
                    & attention_mask[..., None]
                )[:, None, ...]  # Make attention mask 4D

                # For encoder attention only, do not globally expose the final
                # context token. Keep the original context_mask intact for loss.
                encoder_context_mask = context_mask
                has_context = context_mask.any(dim=-1)
                if has_context.any():
                    encoder_context_mask = context_mask.clone()
                    last_context_idx = (
                        encoder_context_mask.long().cumsum(dim=-1)
                        == encoder_context_mask.sum(dim=-1, keepdim=True)
                    ) & encoder_context_mask.bool()
                    encoder_context_mask[last_context_idx & has_context[:, None]] = 0
                
                encoder_attention_mask = (
                    (
                        self.encoder_static_attention_mask[None, ...]
                        | encoder_context_mask[:, None, :]
                    )
                    # self.encoder_static_attention_mask[None, ...]  # used for strictly causal encoder masking
                    & attention_mask[:, None, :]
                    & attention_mask[..., None]
                )[:, None, ...]  # Make attention mask 4D
                encoder_attention_mask = self._preprocess_attention_mask(
                    encoder_attention_mask, dtype=torch.float
                )
                decoder_attention_mask = self._preprocess_attention_mask(
                    decoder_attention_mask, dtype=torch.float
                )
        elif self.config.attn_backend == "flex_attention":
            if context_mask.any():
                raise NotImplementedError(
                    "flex_attention with context_mask not implemented yet."
                )
            elif attention_mask is not None and (attention_mask != 1).any():
                padding_mask = create_attn_mask(attention_mask.bool())
                dec_padding_mask = create_attn_mask(attention_mask.repeat(1, 2).bool())
                enc_masks = [
                    partial(
                        self._encoder_block_mask,
                        block_size=self.config.block_size
                        if self.training
                        else self.config.eval_block_size,
                    ),
                    padding_mask,
                ]
                encoder_attention_mask = create_block_mask(
                    and_masks(*enc_masks),
                    B=input_ids.shape[0],
                    H=None,
                    Q_LEN=input_ids.shape[1],
                    KV_LEN=input_ids.shape[1],
                )
                dec_masks = [
                    partial(
                        self._decoder_block_mask,
                        block_size=self.config.block_size
                        if self.training
                        else self.config.eval_block_size,
                        seq_length=input_ids.shape[1],
                    ),
                    dec_padding_mask,
                ]
                decoder_attention_mask = create_block_mask(
                    and_masks(*dec_masks),
                    B=input_ids.shape[0],
                    H=None,
                    Q_LEN=input_ids.shape[1],
                    KV_LEN=input_ids.shape[1] * 2,
                )
            else:
                encoder_attention_mask = self.encoder_static_attention_mask
                decoder_attention_mask = self.static_attention_mask
        else:
            raise ValueError("Unknown backbone backend")
            
        position_ids = torch.arange(input_ids.shape[1]).to(input_ids.device)[None, :]
        
        if self.training and self.config.train_on_context:
            tokens_mask = attention_mask
        else:
            tokens_mask = attention_mask * (1 - context_mask)
            
        return DenoiserInput(
            xt=xt,
            x0=input_ids,
            attention_mask=decoder_attention_mask,
            tokens_mask=tokens_mask,
            t=None,  # No time for AR
            alpha_t=None,  # No noise schedule
            alpha_t_prime=None,  # No noise schedule
            backbone_kwargs={
                "encoder_input_ids": input_ids,
                "encoder_attention_mask": encoder_attention_mask,
                "encoder_position_ids": position_ids,
                "encoder_cache_position": position_ids[0],
            },
        )

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        """Standard autoregressive cross-entropy loss."""
        decoder_loss_lambda = float(getattr(self.config, "decoder_loss_lambda", 1.0))
        if decoder_loss_lambda <= -1.0:
            raise ValueError(
                "`decoder_loss_lambda` must be greater than -1.0 when using "
                "weighted-average loss normalization."
            )

        def _flatten_for_cross_entropy(
            in_logits: torch.FloatTensor,
            in_targets: torch.LongTensor,
            in_mask: Optional[torch.FloatTensor],
        ) -> tuple[torch.FloatTensor, torch.LongTensor]:
            flat_logits_local = in_logits.contiguous().view(-1, in_logits.size(-1))
            flat_targets_local = in_targets.contiguous().view(-1)
            if in_mask is not None:
                flat_mask_local = in_mask.contiguous().view(-1).bool()
                flat_logits_local = flat_logits_local[flat_mask_local]
                flat_targets_local = flat_targets_local[flat_mask_local]
            return flat_logits_local, flat_targets_local

        # Shift targets: predict next token
        targets = denoiser_inputs.x0[:, 1:]  # Remove first token (typically BOS)
        other_loss_terms = {}
        
        seq_len = denoiser_inputs.x0.shape[1]
        
        # Check if we have encoder logits as well
        if model_output.shape[1] == 2 * seq_len:
            # Encoder logits: first seq_len tokens
            # Decoder logits: last seq_len tokens
            enc_logits = model_output[:, :seq_len-1, :]
            dec_logits = model_output[:, seq_len:-1, :]

            # Duplicate mask if exists
            if denoiser_inputs.tokens_mask is not None:
                M = denoiser_inputs.tokens_mask
                M_loss = torch.cat([M[:, 1:], M[:, 1:]], dim=1)

                # Update denoiser_inputs.tokens_mask for metrics.py
                # Prepend dummy to match the slicing logic in metrics.py
                dummy = torch.zeros((M.shape[0], 1), device=M.device, dtype=M.dtype)
                denoiser_inputs.tokens_mask = torch.cat([dummy, M_loss], dim=1)

                enc_dec_mask = M[:, 1:]
                mask = M_loss
            else:
                enc_dec_mask = None
                mask = None

            enc_flat_logits, enc_flat_targets = _flatten_for_cross_entropy(
                enc_logits,
                targets,
                enc_dec_mask,
            )
            dec_flat_logits, dec_flat_targets = _flatten_for_cross_entropy(
                dec_logits,
                targets,
                enc_dec_mask,
            )
            encoder_loss = torch.nn.functional.cross_entropy(
                enc_flat_logits, enc_flat_targets
            )
            decoder_loss = torch.nn.functional.cross_entropy(
                dec_flat_logits, dec_flat_targets
            )
            loss = (encoder_loss + decoder_loss_lambda * decoder_loss) / (
                1.0 + decoder_loss_lambda
            )
            # Report the raw components, including draft loss when lambda is zero.
            other_loss_terms = {
                "next_token_loss": encoder_loss.detach(),
                "draft_loss": decoder_loss.detach(),
            }

            logits = torch.cat([enc_logits, dec_logits], dim=1)
            targets = torch.cat([targets, targets], dim=1)
        else:
            logits = model_output[:, :-1, :]     # Remove last prediction
            if denoiser_inputs.tokens_mask is not None:
                mask = denoiser_inputs.tokens_mask[:, 1:]
            else:
                mask = None
        
        # Flatten for cross-entropy
        flat_logits = logits.contiguous().view(-1, logits.size(-1))
        flat_targets = targets.contiguous().view(-1)
        
        # Apply token mask if needed
        if mask is not None:
            flat_mask = mask.contiguous().view(-1)
            flat_logits = flat_logits[flat_mask.bool()]
            flat_targets = flat_targets[flat_mask.bool()]
        
        # Compute cross-entropy loss
        if model_output.shape[1] != 2 * seq_len:
            loss = torch.nn.functional.cross_entropy(flat_logits, flat_targets)
        
        # Compute per-token NLLs for compatibility
        with torch.no_grad():
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_nlls = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            if mask is not None:
                token_nlls = token_nlls * mask
        
        return LossAndNllOutput(
            loss=loss,
            nlls=token_nlls,
            other_loss_terms=other_loss_terms,
        )

    def _forward(
        self,
        backbone_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs,
    ) -> torch.FloatTensor:
        """Forward pass - just return the logits for AR."""
        return backbone_output  # No special processing needed for AR

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.LongTensor] = None,
        generation_config: Optional[DiffusionGenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        disable_pbar: bool = False,
        conf_seg: bool = True,             # dynamically decide block size based on confidence
        track_acc_rate: bool = False,      # dynamically decide block size based on acceptance rate (estimated by EMA)
        tree_attn: bool = True,            # tree attention for drafting
        **kwargs: Any,
    ) -> torch.LongTensor:
        assert not (conf_seg and track_acc_rate), "conf_seg and track_acc_rate cannot both be True."

        if generation_config is None:
            assert getattr(self, "generation_config", None) is not None, "Generation config must be provided."
            generation_config = self.generation_config
        if generation_config.do_sample:
            return self._generate_stochastic(
                inputs, generation_config, logits_processor, stopping_criteria,
                max_length, max_new_tokens, batch_size, device, tokenizer,
                disable_pbar, conf_seg, track_acc_rate, tree_attn, **kwargs,
            )
        if inputs is None:
            inputs = torch.ones((batch_size, 1), device=device) * self.bos_token_id
        if max_length is None:
            max_length = generation_config.max_length if hasattr(generation_config, "max_length") else self.max_length
        if max_new_tokens is None:
            max_new_tokens = generation_config.max_new_tokens if hasattr(generation_config, "max_new_tokens") else max_length - inputs.shape[-1]
        
        batch_size = batch_size if batch_size is not None else inputs.shape[0]
        assert batch_size == 1, "Batched sampling not supported yet"
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        block_size = generation_config.block_size

        accumulated_samples = torch.cat([inputs, torch.full(
            (batch_size, 2 * max_new_tokens), # max_new_tokens would be enough, but we still need some buffer in case draft tokens exceeds that length
            self.mask_token_id,
            device=device,
            dtype=torch.long,
        )], dim=-1)

        use_cuda_events = inputs.device.type == "cuda"
        if use_cuda_events:
            overall_start = torch.cuda.Event(enable_timing=True)
            overall_end = torch.cuda.Event(enable_timing=True)
            overall_start.record()
            draft_events = []
        else:
            import time
            overall_start_t = time.perf_counter()
            total_drafting_time_s = 0.0

        if generation_config.align_inputs_to_blocks:
            inputs_offset = block_size * (inputs.shape[-1] // block_size) if inputs.numel() > 0 else 0
        else:
            inputs_offset = inputs.shape[-1] if inputs.numel() > 0 else 0

        total_generated_tokens = 0
        total_accepted_tokens = 0
        total_accepted_lengths = []
        total_draft_lengths = []
        accept_counts = 0
        draft_position_attempt_counts = [0 for _ in range(block_size)]
        draft_position_accept_counts = [0 for _ in range(block_size)]
        
        if generation_config.use_cache and inputs.numel() > 0:
            cache = self.update_cache(
                inputs=inputs[:, : block_size * (inputs.shape[-1] // block_size)]
                if generation_config.align_inputs_to_blocks
                else inputs[:, :-1],
                cache={},
            )
        else:
            cache = None

        current_idx = inputs_offset
        pbar = tqdm(total=max_new_tokens, desc="Tokens", leave=True, disable=disable_pbar)

        # Debug: Print input prompt
        if tokenizer is not None:
            print("\n" + "="*60)
            print("[DEBUG] Generation started")
            print("="*60)
            print(f"[INPUT] Prompt: {tokenizer.decode(inputs[0])}")
            print(f"[CONFIG] max_new_tokens={max_new_tokens}")
            print(f"[CONFIG] conf_seg={conf_seg}, track_acc_rate={track_acc_rate}, tree_attn={tree_attn}")
            print("="*60 + "\n")

        # Confidence-based Segmentation Configuration
        if conf_seg:
            max_draft_len = 100
            conf_threshold = 0.7
        else:
            max_draft_len = block_size
        
        # Tracking Acceptance Rate Configuration
        if track_acc_rate:
            acc_rate_history = []
            ema_acc_rate = 0.8
            ema_alpha = 0.3
            min_draft_len = 1
            max_draft_len_dynamic = 10
            current_draft_len = min(block_size, max_draft_len_dynamic)

        # Flag indicating whether stopping criteria has been met on drafted tokens
        # (so we need to stop and verify these tokens)
        stopped_and_ready_to_verify = False
        while current_idx < inputs_offset + max_new_tokens:
            start_idx = current_idx
            if track_acc_rate:
                actual_max_draft_len = current_draft_len
            else:
                actual_max_draft_len = max_draft_len
                
            # --- DRAFTING PHASE ---
            if use_cuda_events:
                draft_start = torch.cuda.Event(enable_timing=True)
                draft_start.record()
            else:
                import time
                draft_start_t = time.perf_counter()

            draft_tree_candidates = [] 
            actual_draft_len = 0
            for step in range(actual_max_draft_len):
                cur_pos = start_idx + step

                # Decoder input token: use previous token
                dec_inp = accumulated_samples[:, cur_pos - 1 : cur_pos]
                context = accumulated_samples[:, :start_idx] if not generation_config.use_cache else None

                if tokenizer is not None:
                    print(f"[DRAFT] Input Token: '{tokenizer.decode(dec_inp[0].tolist())}' at position {cur_pos}")

                denoiser_inputs, _ = self._prepare_inputs_inference(
                    input_ids=dec_inp,
                    context=context,
                    cache=cache if generation_config.use_cache else None,
                )
                    
                backbone_output = self._backbone_forward(
                    denoiser_inputs,
                    fix_cache_length=True,
                    truncate_cache=False,
                    **cache,
                    **kwargs,
                )
                total_generated_tokens += 1
                backbone_output = {k: v for k, v in backbone_output.items()}
                logits = backbone_output.pop("logits")
                cache = cache | backbone_output

                if logits.ndim == 3:
                    step_logits = logits[:, -1, :]
                else:
                    step_logits = logits

                if logits_processor is not None:
                    step_logits = logits_processor(input_ids=accumulated_samples[:, :cur_pos], scores=step_logits)

                probs = torch.softmax(step_logits, dim=-1)
                top1_prob = probs.max(dim=-1).values.item()
                
                next_tok = step_logits.argmax(dim=-1)
                accumulated_samples[:, cur_pos:cur_pos+1] = next_tok.reshape(batch_size, 1)
                actual_draft_len += 1

                if tokenizer is not None:  # Useful for debugging
                    drafted_token = tokenizer.decode([next_tok.item()])
                    print(f"[DRAFT] Step {step}: token='{drafted_token}' (conf={top1_prob:.3f})")

                if tree_attn:
                    # choose the top-k draft tokens we want to keep for current step
                    if top1_prob > 0.95:
                        k_dynamic = 1
                    elif top1_prob > 0.8:
                        k_dynamic = 3
                    elif top1_prob > 0.5:
                        k_dynamic = 5
                    else:
                        k_dynamic = 10
                    
                    if k_dynamic > 1:
                        _, topk_indices = torch.topk(step_logits, k_dynamic, dim=-1) 
                        topk_ids = topk_indices.flatten().tolist()
                        greedy_id = next_tok.item()
                        
                        if tokenizer is not None:
                            topk_decoded = [tokenizer.decode([tid]) for tid in topk_ids if tid != greedy_id]
                            print(f"[DRAFT]   -> Tree candidates at step {step}: {topk_decoded}")
                        
                        for tid in topk_ids:
                            if tid != greedy_id:
                                draft_tree_candidates.append({
                                    "step": step, 
                                    "token_id": tid,
                                    "pos_id": cur_pos
                                })

                if stopping_criteria is not None:
                    is_done = stopping_criteria(
                        input_ids=accumulated_samples[:, inputs_offset : cur_pos + 1],
                        scores=None,
                    )
                    if torch.any(is_done):
                        stopped_and_ready_to_verify = True
                        break
                
                if conf_seg and top1_prob < conf_threshold:
                    break
            
            # Debug: Print draft summary
            if tokenizer is not None:
                drafted_tokens = accumulated_samples[0, start_idx:start_idx + actual_draft_len]
                drafted_text = tokenizer.decode(drafted_tokens)
                print(f"\n[DRAFT COMPLETE] Drafted {actual_draft_len} tokens: '{drafted_text}'")
                if tree_attn and len(draft_tree_candidates) > 0:
                    print(f"[DRAFT TREE] {len(draft_tree_candidates)} alternative candidates in tree")
                print("-" * 40)

            # --- PRE-VERIFICATION CACHE TRUNCATION ---
            draft_len_to_truncate = actual_draft_len if (conf_seg or track_acc_rate) else block_size
            num_layers = len(cache['past_key_values'].key_cache)
            is_share_kv = (self.backbone.encoder == self.backbone.decoder)
            for layer in range(num_layers):
                if not is_share_kv or layer >= self.backbone.decoder_layer_idxs[0]:
                    cache['past_key_values'].key_cache[layer] = cache['past_key_values'].key_cache[layer][..., :-draft_len_to_truncate, :]
                    cache['past_key_values'].value_cache[layer] = cache['past_key_values'].value_cache[layer][..., :-draft_len_to_truncate, :]

            if use_cuda_events:
                draft_end = torch.cuda.Event(enable_timing=True)
                draft_end.record()
                draft_events.append((draft_start, draft_end))
            else:
                import time
                total_drafting_time_s += time.perf_counter() - draft_start_t

            # --- VERIFICATION PHASE ---
            if generation_config.use_cache:
                num_generated = actual_draft_len
                
                greedy_inputs = accumulated_samples[:, start_idx - 1 : start_idx + num_generated]
                greedy_len = greedy_inputs.shape[1]
                
                if tree_attn and len(draft_tree_candidates) > 0:
                    extra_tokens_list = [c["token_id"] for c in draft_tree_candidates]
                    extra_inputs = torch.tensor([extra_tokens_list], device=device, dtype=torch.long)
                    encoder_inputs = torch.cat([greedy_inputs, extra_inputs], dim=1)
                    
                    past_len = cache['past_key_values'].key_cache[0].shape[-2]
                    start_pos = past_len 
                    greedy_pos_ids = torch.arange(start_pos, start_pos + greedy_len, device=device).unsqueeze(0)
                    extra_pos_ids = torch.tensor([[start_pos + 1 + c["step"] for c in draft_tree_candidates]], device=device)
                    position_ids = torch.cat([greedy_pos_ids, extra_pos_ids], dim=1)

                    # Attention Mask
                    total_len = encoder_inputs.shape[1]
                    total_past = past_len
                    mask = torch.full((1, 1, total_len, total_past + total_len), torch.finfo(torch.float32).min, device=device)
                    mask[:, :, :, :total_past] = 0.0
                    
                    causal_mask = torch.triu(torch.full((greedy_len, greedy_len), float('-inf'), device=device), diagonal=1)
                    mask[:, :, :greedy_len, total_past:total_past+greedy_len] = causal_mask.unsqueeze(0).unsqueeze(0)
                    mask[:, :, :greedy_len, total_past:total_past+greedy_len].masked_fill_(
                        torch.tril(torch.ones((greedy_len, greedy_len), device=device)).bool(), 0.0
                    )
                    
                    for idx, cand in enumerate(draft_tree_candidates):
                        row_idx = greedy_len + idx
                        step = cand["step"]
                        mask[:, :, row_idx, total_past + row_idx] = 0.0
                        mask[:, :, row_idx, total_past : total_past + 1 + step] = 0.0
                    
                    mask = mask.to(self.backbone.encoder.dtype)
                else:
                    encoder_inputs = greedy_inputs
                    position_ids = None
                    mask = None

                # --- FIX: Handling Argument Collision ---
                
                # 1. Clean kwargs of explicit overrides
                safe_kwargs = kwargs.copy()
                if position_ids is not None: safe_kwargs.pop("position_ids", None)
                if mask is not None: safe_kwargs.pop("attention_mask", None)

                # 2. Create a temporary cache dict to pass to _prepare_inputs_inference.
                #    This is crucial because _prepare will POP 'past_key_values'.
                #    We want it to pop from this temp object, so when we pass **temp_cache to forward,
                #    it is empty of KVs, preventing collision.
                temp_cache = cache.copy() if cache is not None else {}
                if position_ids is not None: temp_cache.pop("position_ids", None)
                if mask is not None: temp_cache.pop("attention_mask", None)

                # 3. Prepare Overrides
                prepare_overrides = {}
                if position_ids is not None: prepare_overrides["encoder_position_ids"] = position_ids
                if mask is not None: prepare_overrides["encoder_attention_mask"] = mask

                # 4. Prepare inputs (Consumes KVs from temp_cache)
                context_input, _ = self._prepare_inputs_inference(
                    input_ids=encoder_inputs,
                    cache=temp_cache, 
                    return_updated_cache=True,
                    **prepare_overrides, 
                    **safe_kwargs
                )

                backbone_args = {
                    "return_updated_cache": True,
                    "return_last_hidden_state": True,
                    "enforce_causal_mask": (mask is None),
                    "preserve_encoder_attention_mask": (mask is not None),
                }
                
                # 5. Forward Pass (Uses temp_cache which no longer has KVs)
                backbone_output = self._backbone_forward(
                    context_input,
                    **backbone_args,
                    **temp_cache, 
                    **safe_kwargs
                )
                
                # ... (Rest of logic remains identical) ...
                encoder_hidden = backbone_output['last_hidden_state']
                encoder_logits = self.backbone.encoder.lm_head(encoder_hidden)
                
                greedy_logits = encoder_logits[:, :greedy_len]
                draft_tokens = greedy_inputs[:, 1:] 
                pred_logits = greedy_logits[:, :-1]
                pred_tokens = pred_logits.argmax(dim=-1)
                
                matches = (draft_tokens == pred_tokens)
                valid_mask = matches.cumprod(dim=1)
                min_accepted = valid_mask.sum(dim=1).min().item()
                
                # Debug: Print verification results
                if tokenizer is not None:
                    print(f"[VERIFY] Checking {draft_tokens.shape[1]} drafted tokens...")
                    for i in range(draft_tokens.shape[1]):
                        draft_tok = tokenizer.decode([draft_tokens[0, i].item()])
                        pred_tok = tokenizer.decode([pred_tokens[0, i].item()])
                        match_status = "✓" if matches[0, i].item() else "✗"
                        print(f"[VERIFY]   Position {i}: drafted='{draft_tok}' vs verified='{pred_tok}' [{match_status}]")
                    print(f"[VERIFY] Accepted {min_accepted}/{draft_tokens.shape[1]} tokens")
                
                switched_to_branch = False
                branch_correction_token = None
                branch_index_in_extra = -1
                
                if tree_attn and min_accepted < actual_draft_len and len(draft_tree_candidates) > 0:
                    target_token_at_mismatch = pred_tokens[0, min_accepted].item()
                    step_of_mismatch = min_accepted 
                    
                    for i, cand in enumerate(draft_tree_candidates):
                        if cand["step"] == step_of_mismatch and cand["token_id"] == target_token_at_mismatch:
                            switched_to_branch = True
                            branch_index_in_extra = i
                            branch_logits = encoder_logits[0, greedy_len + i]
                            branch_correction_token = branch_logits.argmax().item()
                            break
                
                if switched_to_branch:
                    mismatch_pos = start_idx + min_accepted
                    accumulated_samples[:, mismatch_pos] = draft_tree_candidates[branch_index_in_extra]["token_id"]
                    
                    # Debug: Print branch switch info
                    if tokenizer is not None:
                        branch_tok = tokenizer.decode([draft_tree_candidates[branch_index_in_extra]["token_id"]])
                        correction_tok = tokenizer.decode([branch_correction_token])
                        print(f"[TREE BRANCH] Switched to tree branch at position {min_accepted}!")
                        print(f"[TREE BRANCH]   Branch token: '{branch_tok}'")
                        print(f"[TREE BRANCH]   Correction (next) token: '{correction_tok}'")

                    if stopping_criteria is not None:
                        is_done = stopping_criteria(input_ids=accumulated_samples[:, inputs_offset : mismatch_pos + 1], scores=None)
                        if torch.any(is_done):
                            accumulated_samples = accumulated_samples[:, : mismatch_pos + 1]
                            break

                    accumulated_samples[:, mismatch_pos + 1] = branch_correction_token

                    if stopping_criteria is not None:
                        is_done = stopping_criteria(
                            input_ids=accumulated_samples[
                                :, inputs_offset : mismatch_pos + 2
                            ],
                            scores=None,
                        )
                        if torch.any(is_done):
                            accumulated_samples = accumulated_samples[
                                :, : mismatch_pos + 2
                            ]
                            break
                    
                    total_accepted_tokens += (min_accepted + 1)
                    final_accepted_len = min_accepted + 1
                    total_accepted_lengths.append(final_accepted_len)
                    total_draft_lengths.append(actual_draft_len)
                    accept_counts += 1
                    
                    indices_to_keep = list(range(min_accepted + 1)) 
                    indices_to_keep.append(greedy_len + branch_index_in_extra)
                    indices_tensor = torch.tensor(indices_to_keep, device=device)
                    
                    new_kv = backbone_output['past_key_values']
                    for layer in range(len(new_kv)):
                        k = new_kv.key_cache[layer]
                        v = new_kv.value_cache[layer]
                        
                        total_hist_len = k.shape[-2]
                        new_len = encoder_inputs.shape[1]
                        past_len_val = total_hist_len - new_len
                        
                        k_past = k[..., :past_len_val, :]
                        v_past = v[..., :past_len_val, :]
                        
                        k_new = k[..., past_len_val:, :].index_select(-2, indices_tensor)
                        v_new = v[..., past_len_val:, :].index_select(-2, indices_tensor)
                        
                        new_kv.key_cache[layer] = torch.cat([k_past, k_new], dim=-2)
                        new_kv.value_cache[layer] = torch.cat([v_past, v_new], dim=-2)
                    
                    cache['past_key_values'] = new_kv

                else:
                    total_accepted_tokens += min_accepted
                    total_accepted_lengths.append(min_accepted)
                    total_draft_lengths.append(actual_draft_len)
                    accept_counts += 1
                    correction_token = greedy_logits[:, min_accepted].argmax(dim=-1)
                    
                    # Debug: Print correction token info
                    if tokenizer is not None:
                        correction_tok = tokenizer.decode([correction_token.item()])
                        print(f"[GREEDY PATH] Accepted {min_accepted} tokens, correction token: '{correction_tok}'")

                    if stopped_and_ready_to_verify and stopping_criteria is not None and min_accepted > 0:
                        is_done = stopping_criteria(input_ids=accumulated_samples[:, inputs_offset : start_idx + min_accepted], scores=None)
                        if torch.any(is_done):
                            accumulated_samples = accumulated_samples[:, : start_idx + min_accepted]
                            break

                    accumulated_samples[:, start_idx + min_accepted] = correction_token

                    if stopping_criteria is not None:
                        is_done = stopping_criteria(
                            input_ids=accumulated_samples[
                                :, inputs_offset : start_idx + min_accepted + 1
                            ],
                            scores=None,
                        )
                        if torch.any(is_done):
                            accumulated_samples = accumulated_samples[
                                :, : start_idx + min_accepted + 1
                            ]
                            break

                    final_accepted_len = min_accepted
                    
                    target_len = start_idx + min_accepted
                    enc_kv = cache['past_key_values'] if 'past_key_values' in cache else backbone_output['past_key_values']
                    
                    for layer in range(len(enc_kv)):
                        enc_kv.key_cache[layer] = enc_kv.key_cache[layer][..., :target_len, :]
                        enc_kv.value_cache[layer] = enc_kv.value_cache[layer][..., :target_len, :]
                    
                    cache['past_key_values'] = enc_kv

                accepted_prefix_len = min(final_accepted_len, actual_draft_len)
                for pos in range(actual_draft_len):
                    if pos >= len(draft_position_attempt_counts):
                        draft_position_attempt_counts.append(0)
                        draft_position_accept_counts.append(0)
                    draft_position_attempt_counts[pos] += 1
                    if pos < accepted_prefix_len:
                        draft_position_accept_counts[pos] += 1

                if track_acc_rate:
                    acc_rate_history.append((min_accepted, actual_draft_len))
                    window_acc_count = 0
                    window_draft_count = 0
                    target_window_tokens = 20
                    for hist_acc, hist_draft in reversed(acc_rate_history):
                        window_acc_count += hist_acc
                        window_draft_count += hist_draft
                        if window_draft_count >= target_window_tokens: break
                    
                    current_window_acc_rate = window_acc_count / window_draft_count if window_draft_count > 0 else 0.0
                    ema_acc_rate = ema_alpha * current_window_acc_rate + (1 - ema_alpha) * ema_acc_rate
                    
                    if ema_acc_rate > 0.8:
                        current_draft_len = min(current_draft_len + 1, max_draft_len_dynamic)
                    elif ema_acc_rate < 0.7:
                        current_draft_len = max(current_draft_len - 1, min_draft_len)

                advance = final_accepted_len + 1
                current_idx += advance
                pbar.update(advance)
                
                # Debug: Print current generation state
                if tokenizer is not None:
                    current_output = accumulated_samples[0, inputs_offset:current_idx]
                    current_text = tokenizer.decode(current_output)
                    print(f"\n[PROGRESS] Generated so far ({current_idx - inputs_offset} tokens): '{current_text}'")
                    print("=" * 60 + "\n")

                if stopped_and_ready_to_verify and stopping_criteria is not None:
                    is_done = stopping_criteria(input_ids=accumulated_samples[:, inputs_offset : current_idx], scores=None)
                    if torch.any(is_done):
                        accumulated_samples = accumulated_samples[:, : current_idx]
                        break
                    else:
                        stopped_and_ready_to_verify = False

            else:
                cache = self.update_cache(
                    inputs=accumulated_samples[:, start_idx-1:start_idx + block_size-1],
                    cache=cache,
                )
                current_idx += block_size
                pbar.update(block_size)

        # Debug: Print final summary
        if tokenizer is not None:
            final_output = accumulated_samples[0, inputs_offset:]
            # Remove trailing mask tokens if any
            mask_positions = (final_output == self.mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_positions) > 0:
                final_output = final_output[:mask_positions[0]]
            final_text = tokenizer.decode(final_output)
            acceptance_rate = total_accepted_tokens / total_generated_tokens if total_generated_tokens > 0 else 0
            print("\n" + "=" * 60)
            print("[DEBUG] SEED.generate() completed")
            print("=" * 60)
            print(f"[FINAL OUTPUT]: '{final_text}'")
            print(f"[STATS] Total forward passes: {total_generated_tokens}")
            print(f"[STATS] Total accepted tokens: {total_accepted_tokens}")
            print(f"[STATS] Acceptance rate: {acceptance_rate:.2%}")
            avg_draft_len = (
                sum(total_draft_lengths) / len(total_draft_lengths)
                if len(total_draft_lengths) > 0
                else 0.0
            )
            print(f"[STATS] Avg draft length: {avg_draft_len:.2f}")
            print("=" * 60 + "\n")

        self._last_draft_position_acceptance = {
            "attempt_counts": draft_position_attempt_counts,
            "accept_counts": draft_position_accept_counts,
            "draft_lengths": total_draft_lengths,
            "acceptance_rates": [
                (acc / att) if att > 0 else 0.0
                for acc, att in zip(draft_position_accept_counts, draft_position_attempt_counts)
            ],
        }

        if use_cuda_events:
            overall_end.record()
            torch.cuda.synchronize()
            total_all_time_s = overall_start.elapsed_time(overall_end) / 1000.0
            total_drafting_time_s = sum([s.elapsed_time(e) for s, e in draft_events]) / 1000.0
        else:
            import time
            total_all_time_s = time.perf_counter() - overall_start_t

        return accumulated_samples, (total_generated_tokens, total_accepted_tokens), (total_accepted_lengths, accept_counts), (total_drafting_time_s, total_all_time_s)
