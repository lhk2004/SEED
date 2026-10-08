import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (
    DynamicCache,
    GenerationConfig,
    LogitsProcessorList,
    PreTrainedTokenizer,
    StoppingCriteriaList,
)
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.denoiser.ar import AR, ARConfig
from src.denoiser.base import (
    DenoiserInput,
    DenoiserOutput,
    LossAndNllOutput,
)


class LayerSkipConfig(ARConfig):
    """Configuration class for LayerSkip models.

    LayerSkip trains a standard autoregressive model with an auxiliary
    early-exit loss: at each training step, a random intermediate layer
    is selected, its hidden state is projected through the final norm
    and lm_head, and a cross-entropy loss is computed. This early-exit
    loss is combined with the standard last-layer loss.

    At inference time, the early layers can be used as a draft model for
    self-speculative decoding (via HuggingFace's `assistant_early_exit`).
    """

    model_type = "layerskip"

    def __init__(
        self,
        length: Optional[int] = None,
        backbone_config: Optional[Dict[str, Any]] = None,
        tokenization_config: Optional[Dict[str, Any]] = None,
        noise_config: None = None,
        time_conditioned_backbone: Optional[bool] = None,
        early_exit_scale: float = 1.0,
        early_exit_loss_scale: Optional[float] = None,
        early_exit_curriculum: str = "none",
        early_exit_rotation_stride: int = 1,
        layer_dropout_max: float = 0.0,
        layer_dropout_time_schedule: str = "constant",
        layer_dropout_total_steps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            length=length,
            backbone_config=backbone_config,
            noise_config=noise_config,
            tokenization_config=tokenization_config,
            time_conditioned_backbone=time_conditioned_backbone,
            **kwargs,
        )
        if early_exit_loss_scale is not None:
            early_exit_scale = early_exit_loss_scale
        self.early_exit_scale = early_exit_scale
        self.early_exit_curriculum = early_exit_curriculum
        self.early_exit_rotation_stride = early_exit_rotation_stride
        self.layer_dropout_max = layer_dropout_max
        self.layer_dropout_time_schedule = layer_dropout_time_schedule
        self.layer_dropout_total_steps = layer_dropout_total_steps


class LayerSkip(AR):
    """LayerSkip: autoregressive model with early-exit auxiliary loss.

    During training, at each step:
      1. A random layer index in [1, num_hidden_layers - 1] is selected.
      2. The hidden state at that layer is projected through the model's
         final RMSNorm and lm_head to produce early-exit logits.
      3. A cross-entropy loss is computed from these early-exit logits.
      4. The final loss is a weighted combination of the last-layer loss
         and the early-exit loss.

    Reference: https://arxiv.org/abs/2404.16710
    """

    config_class = LayerSkipConfig

    def __init__(
        self,
        config: LayerSkipConfig,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self._train_step_counter = 0
        self.early_exit_scale = getattr(config, "early_exit_scale", 1.0)
        self.early_exit_curriculum = getattr(config, "early_exit_curriculum", "none")
        self.early_exit_rotation_stride = max(
            1, int(getattr(config, "early_exit_rotation_stride", 1))
        )
        self.layer_dropout_max = float(getattr(config, "layer_dropout_max", 0.0))
        self.layer_dropout_time_schedule = getattr(
            config, "layer_dropout_time_schedule", "constant"
        )
        self.layer_dropout_total_steps = getattr(config, "layer_dropout_total_steps", None)

    def _get_train_step(self) -> int:
        return self._train_step_counter

    def _get_layer_dropout_time_scale(self, step: int) -> float:
        if self.layer_dropout_time_schedule == "constant":
            return 1.0
        if self.layer_dropout_time_schedule == "exponential":
            total_steps = self.layer_dropout_total_steps
            if total_steps is None or total_steps <= 1:
                return 1.0
            t = min(max(step, 0), total_steps - 1)
            return math.exp((t * math.log(2.0)) / (total_steps - 1)) - 1.0
        raise ValueError(
            "layer_dropout_time_schedule must be one of {'constant', 'exponential'}"
        )

    @staticmethod
    def _get_layer_dropout_depth_scale(layer_idx: int, num_layers: int) -> float:
        if num_layers <= 1:
            return 0.0
        return math.exp((layer_idx * math.log(2.0)) / (num_layers - 1)) - 1.0

    def _get_layer_dropout_prob(self, layer_idx: int, num_layers: int, step: int) -> float:
        prob = (
            self.layer_dropout_max
            * self._get_layer_dropout_time_scale(step)
            * self._get_layer_dropout_depth_scale(layer_idx, num_layers)
        )
        return float(max(0.0, min(1.0, prob)))

    def _get_early_exit_curriculum_mask(self, num_layers: int, step: int) -> torch.BoolTensor:
        if self.early_exit_curriculum == "none":
            return torch.ones(num_layers, dtype=torch.bool)
        if self.early_exit_curriculum == "rotational":
            stride = max(1, self.early_exit_rotation_stride)
            offset = step % stride
            layer_ids = torch.arange(num_layers)
            return ((layer_ids - offset) % stride) == 0
        raise ValueError(
            "early_exit_curriculum must be one of {'none', 'rotational'}"
        )

    def _layer_dropout_backbone_forward(
        self,
        denoiser_inputs: DenoiserInput,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        model = self.backbone.model
        hf_model = model.model

        input_ids = denoiser_inputs.xt
        attention_mask = denoiser_inputs.attention_mask
        hidden_states = hf_model.embed_tokens(input_ids)

        batch_size, seq_len = hidden_states.shape[:2]
        cache_position = torch.arange(seq_len, device=input_ids.device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        causal_mask = hf_model._update_causal_mask(
            attention_mask,
            hidden_states,
            cache_position,
            None,
            output_attentions=False,
        )

        all_hidden_states = [hidden_states] if output_hidden_states else None
        step = self._get_train_step()
        layers = hf_model.layers
        num_layers = len(layers)

        for layer_idx, layer in enumerate(layers):
            drop_prob = self._get_layer_dropout_prob(layer_idx, num_layers, step)
            keep_mask = (
                torch.rand(batch_size, device=hidden_states.device) > drop_prob
                if drop_prob > 0.0
                else torch.ones(batch_size, device=hidden_states.device, dtype=torch.bool)
            )

            if keep_mask.any():
                if keep_mask.all():
                    in_states = hidden_states
                    in_position_ids = position_ids
                    in_attention_mask = causal_mask
                else:
                    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
                    in_states = hidden_states[keep_idx]
                    in_position_ids = position_ids[keep_idx]
                    if causal_mask is not None and causal_mask.shape[0] == batch_size:
                        in_attention_mask = causal_mask[keep_idx]
                    else:
                        in_attention_mask = causal_mask

                in_position_embeddings = hf_model.rotary_emb(in_states, in_position_ids)
                layer_outputs = layer(
                    in_states,
                    attention_mask=in_attention_mask,
                    position_ids=in_position_ids,
                    past_key_value=None,
                    cache_position=cache_position,
                    position_embeddings=in_position_embeddings,
                )
                out_states = layer_outputs[0]

                if keep_mask.all():
                    hidden_states = out_states
                else:
                    hidden_states = hidden_states.clone()
                    hidden_states[keep_idx] = out_states

            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        final_hidden = hf_model.norm(hidden_states)
        logits = model.lm_head(final_hidden)

        return CausalLMOutputWithPast(
            logits=logits,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            past_key_values=None,
        )

    def _get_num_hidden_layers(self) -> int:
        """Get the number of hidden layers in the backbone model."""
        return len(self.backbone.model.model.layers)

    def _get_norm(self):
        """Get the final normalization layer from the backbone."""
        return self.backbone.model.model.norm

    def _get_lm_head(self):
        """Get the lm_head from the backbone."""
        return self.backbone.model.lm_head

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        compute_loss: Optional[bool] = True,
        **kwargs,
    ) -> DenoiserOutput:
        """Forward pass with early-exit loss for LayerSkip training.

        Overrides the base forward to:
        1. Pass output_hidden_states=True to the backbone.
        2. Extract hidden states for early-exit loss computation.
        3. Combine early-exit loss with the standard last-layer loss.
        """
        denoiser_inputs = self._prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_mask=context_mask,
            past_key_values=past_key_values,
            t=t,
        )
        need_hidden_states = bool(self.training and compute_loss)

        if (
            self.training
            and self.layer_dropout_max > 0.0
            and past_key_values is None
        ):
            backbone_output = self._layer_dropout_backbone_forward(
                denoiser_inputs,
                output_hidden_states=need_hidden_states,
                **kwargs,
            )
        else:
            backbone_output = self._backbone_forward(
                denoiser_inputs,
                output_hidden_states=need_hidden_states,
                **kwargs,
            )

        new_past_key_values = getattr(backbone_output, "past_key_values", None)
        hidden_states = getattr(backbone_output, "hidden_states", None)
        backbone_logits = getattr(backbone_output, "logits", backbone_output[0])

        denoiser_output = self._forward(
            backbone_logits,
            denoiser_inputs,
            **kwargs,
        )

        if compute_loss:
            loss_and_nll = self._compute_loss(
                model_output=denoiser_output,
                denoiser_inputs=denoiser_inputs,
                hidden_states=hidden_states,
                **kwargs,
            )
            loss = loss_and_nll.loss
            nlls = loss_and_nll.nlls
            other_loss_terms = loss_and_nll.other_loss_terms
        else:
            loss, nlls = None, None
            other_loss_terms = {}

        if self.training:
            self._train_step_counter += 1

        return DenoiserOutput(
            denoiser_output=denoiser_output,
            logits=backbone_logits,
            past_key_values=new_past_key_values,
            tokens_mask=denoiser_inputs.tokens_mask,
            loss=loss,
            nlls=nlls,
            other_loss_terms=other_loss_terms,
        )

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        """Compute combined last-layer + early-exit loss.

        Args:
            model_output: Logits from the last layer (B, L, V).
            denoiser_inputs: Standard denoiser inputs with x0, tokens_mask, etc.
            hidden_states: Tuple of hidden states from all layers.
                hidden_states[0] = embedding output,
                hidden_states[i] = output of layer i (1-indexed).
                hidden_states[-1] = output of the last layer (after norm in some
                    implementations, but typically before norm for HF models).
        """
        # --- Last-layer loss (standard AR cross-entropy) ---
        # AR's _prepare_inputs already shifts: x0 = input_ids[1:] with shape (B, L-1, 1)
        # and xt = input_ids[:-1] with shape (B, L-1). No further shifting needed.
        targets = denoiser_inputs.x0.squeeze(-1)  # (B, L-1)
        log_probs = model_output  # (B, L-1, V)
        mask = denoiser_inputs.tokens_mask  # (B, L-1) or None
        if mask is None:
            mask = torch.ones_like(targets, dtype=log_probs.dtype)

        def _token_ce_from_log_probs(layer_log_probs: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
            token_nlls = -torch.gather(layer_log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
            masked_nlls = token_nlls * mask
            count = mask.sum(dim=-1).clamp_min(1.0)
            token_nll = (masked_nlls.sum(dim=-1) / count).mean()
            return token_nll, masked_nlls

        loss_last, token_nlls = _token_ce_from_log_probs(log_probs)

        # --- Early-exit loss ---
        other_loss_terms = {}

        if self.training and hidden_states is not None:
            num_hidden_layers = self._get_num_hidden_layers()
            norm = self._get_norm()
            lm_head = self._get_lm_head()

            if hidden_states is None or len(hidden_states) < (num_hidden_layers + 1):
                loss = loss_last
            else:
                early_losses: List[torch.Tensor] = []
                raw_weights: List[float] = []

                step = self._get_train_step()
                curriculum_mask = self._get_early_exit_curriculum_mask(num_hidden_layers, step).to(
                    device=log_probs.device
                )

                triangular_prev_sum = (num_hidden_layers - 2) * (num_hidden_layers - 1) / 2

                for layer_idx in range(num_hidden_layers):
                    if not bool(curriculum_mask[layer_idx]):
                        continue

                    layer_hidden = hidden_states[layer_idx + 1].to(norm.weight.dtype)
                    layer_logits = lm_head(norm(layer_hidden))
                    layer_log_probs = F.log_softmax(layer_logits, dim=-1)
                    layer_loss, _ = _token_ce_from_log_probs(layer_log_probs)
                    early_losses.append(layer_loss)

                    if layer_idx < num_hidden_layers - 1:
                        raw_weight = self.early_exit_scale * (layer_idx * (layer_idx + 1) / 2)
                    else:
                        raw_weight = (num_hidden_layers - 1) + self.early_exit_scale * triangular_prev_sum
                    raw_weights.append(float(raw_weight))

                if len(early_losses) == 0:
                    loss = loss_last
                else:
                    weights = torch.tensor(
                        raw_weights,
                        dtype=early_losses[0].dtype,
                        device=early_losses[0].device,
                    )
                    if float(weights.sum()) == 0.0:
                        weights = torch.ones_like(weights)
                    weights = weights / weights.sum()

                    stacked_losses = torch.stack(early_losses)
                    loss = (weights * stacked_losses).sum()

                    other_loss_terms["last_layer_loss"] = loss_last.detach()
                    other_loss_terms["early_exit_loss"] = loss.detach()
                    other_loss_terms["early_exit_layers_used"] = int(len(early_losses))
                    other_loss_terms["early_exit_weights_sum"] = weights.sum().detach()
        else:
            loss = loss_last

        return LossAndNllOutput(
            loss=loss,
            nlls=token_nlls,
            other_loss_terms=other_loss_terms,
        )

    def _early_exit_forward(
        self,
        input_ids: torch.LongTensor,
        exit_layer: int,
        past_key_values: Optional[DynamicCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_exit_hidden: bool = False,
    ) -> Union[
        Tuple[torch.FloatTensor, DynamicCache],
        Tuple[torch.FloatTensor, DynamicCache, torch.FloatTensor],
    ]:
        """Run forward pass through only the first `exit_layer` layers.

        Returns logits from the early exit and the KV cache up to that layer.
        """
        model = self.backbone.model
        hf_model = model.model  # The inner transformer (e.g., Qwen3Model)

        # Embedding
        inputs_embeds = hf_model.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        # Set up cache
        if past_key_values is None:
            past_key_values = DynamicCache()

        cache_position = torch.arange(
            past_key_values.get_seq_length() if len(past_key_values) > 0 else 0,
            (past_key_values.get_seq_length() if len(past_key_values) > 0 else 0) + input_ids.shape[1],
            device=input_ids.device,
        )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Prepare causal mask
        causal_mask = model.model._update_causal_mask(
            attention_mask, hidden_states, cache_position, past_key_values, output_attentions=False
        )

        # Compute rotary position embeddings (shared across all layers)
        position_embeddings = hf_model.rotary_emb(hidden_states, position_ids)

        # Run through only the first `exit_layer` layers
        for i, layer in enumerate(hf_model.layers[:exit_layer]):
            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        # Keep pre-norm exit hidden states for verification stage (KVQ-style reuse)
        exit_hidden_states = hidden_states

        # Apply shared LM head for early-exit logits
        logits = model.lm_head(hf_model.norm(exit_hidden_states))

        if return_exit_hidden:
            return logits, past_key_values, exit_hidden_states
        return logits, past_key_values

    def _remaining_layers_forward_from_exit(
        self,
        exit_hidden_states: torch.FloatTensor,
        exit_layer: int,
        past_key_values: Optional[DynamicCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, DynamicCache]:
        """Run only layers [exit_layer, L-1] starting from exit-layer hidden states."""
        model = self.backbone.model
        hf_model = model.model

        hidden_states = exit_hidden_states

        if past_key_values is None:
            past_key_values = DynamicCache()

        past_len = past_key_values.get_seq_length() if len(past_key_values) > 0 else 0
        cache_position = torch.arange(
            past_len,
            past_len + hidden_states.shape[1],
            device=hidden_states.device,
        )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = hf_model._update_causal_mask(
            attention_mask,
            hidden_states,
            cache_position,
            past_key_values,
            output_attentions=False,
        )
        position_embeddings = hf_model.rotary_emb(hidden_states, position_ids)

        for layer in hf_model.layers[exit_layer:]:
            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        hidden_states = hf_model.norm(hidden_states)
        logits = model.lm_head(hidden_states)

        return logits, past_key_values

    def _full_forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[DynamicCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, DynamicCache]:
        """Run a full forward pass through all layers. Returns logits and cache."""
        outputs = self.backbone.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outputs.logits, outputs.past_key_values

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.LongTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        disable_pbar: Optional[bool] = None,
        assistant_early_exit: Optional[int] = None,
        **kwargs,
    ) -> Union[
        torch.LongTensor,
        Tuple[
            torch.LongTensor,
            Tuple[int, int],
            Tuple[List[int], int],
            Tuple[float, float],
        ],
    ]:
        """Generate with optional self-speculative decoding via early exit.

        When `assistant_early_exit` is provided, performs self-speculative
        decoding: the first `assistant_early_exit` layers draft tokens,
        then the full model verifies them. Returns the same 3-tuple as
        SEED.generate() for acceptance metrics tracking.

        When `assistant_early_exit` is None, falls back to standard AR
        generation (single tensor return).

        Args:
            assistant_early_exit: Layer index for drafting. Should be in
                [1, num_hidden_layers - 1]. If None, standard AR generation.

        Returns:
            If assistant_early_exit is None:
                torch.LongTensor: Generated token ids.
            If assistant_early_exit is set:
                Tuple of:
                    - samples (torch.LongTensor): Generated token ids.
                    - (total_generated_tokens, total_accepted_tokens)
                    - (accepted_lengths_list, accept_counts)
        """
        # --- Standard AR generation (no speculative decoding) ---
        if assistant_early_exit is None:
            outputs = self.backbone.model.generate(
                inputs=inputs,
                attention_mask=torch.ones_like(inputs),
                generation_config=generation_config,
                logits_processor=logits_processor,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            if tokenizer is not None:
                print(tokenizer.batch_decode(outputs))
            return outputs

        # --- Self-speculative decoding with acceptance tracking ---
        import time
        overall_start_t = time.perf_counter()
        total_drafting_time_s = 0.0

        if inputs is None:
            inputs = torch.ones((batch_size or 1, 1), device=device, dtype=torch.long) * self.bos_token_id
        batch_size = inputs.shape[0]
        assert batch_size == 1, "Batched self-speculative decoding not supported yet"
        if device is None:
            device = inputs.device
        if max_new_tokens is None:
            if generation_config is not None and hasattr(generation_config, "max_new_tokens") and generation_config.max_new_tokens is not None:
                max_new_tokens = generation_config.max_new_tokens
            elif max_length is not None:
                max_new_tokens = max_length - inputs.shape[-1]
            else:
                max_new_tokens = 512

        num_hidden_layers = self._get_num_hidden_layers()
        assert 1 <= assistant_early_exit < num_hidden_layers, (
            f"assistant_early_exit must be in [1, {num_hidden_layers - 1}], "
            f"got {assistant_early_exit}"
        )

        # Draft length: how many tokens the drafter proposes per round
        # draft_len = assistant_early_exit  # heuristic: more layers -> longer drafts
        # draft_len = max(1, min(draft_len, 16))  # clamp to [1, 16]
        draft_len = 6

        # Metrics tracking
        total_generated_tokens = 0  # drafted tokens proposed by early-exit model
        total_accepted_tokens = 0
        total_accepted_lengths: List[int] = []
        accept_counts = 0
        draft_position_attempt_counts: List[int] = [0 for _ in range(draft_len)]
        draft_position_accept_counts: List[int] = [0 for _ in range(draft_len)]

        # Start with the prompt
        generated = inputs.clone()
        prompt_len = inputs.shape[1]

        # Prefill: run full model on prompt to populate KV cache
        full_cache = DynamicCache()
        if prompt_len > 1:
            prefill_logits, full_cache = self._full_forward(
                input_ids=inputs[:, :-1],
                past_key_values=full_cache,
            )
        # We need the last prompt token to start drafting
        # Get logits for the last prompt token
        last_token = inputs[:, -1:]
        last_logits, full_cache = self._full_forward(
            input_ids=last_token,
            past_key_values=full_cache,
        )

        # The next token from verifier (to start the first draft round)
        next_token = last_logits[:, -1:].argmax(dim=-1)
        generated = torch.cat([generated, next_token], dim=-1)

        pbar = tqdm(
            total=max_new_tokens,
            desc="LayerSkip Generate",
            disable=disable_pbar,
        )
        pbar.update(1)  # first token from prefill

        tokens_generated = 1
        done = False

        while tokens_generated < max_new_tokens and not done:
            # --- DRAFT PHASE: use early layers to draft tokens ---
            draft_start_t = time.perf_counter()
            draft_tokens = []
            verify_input_exit_hiddens = []
            draft_cache = DynamicCache()
            # Copy the full cache up to exit_layer for drafting
            for layer_idx in range(min(assistant_early_exit, len(full_cache))):
                draft_cache.update(
                    full_cache.key_cache[layer_idx].clone(),
                    full_cache.value_cache[layer_idx].clone(),
                    layer_idx,
                )

            current_draft_input = generated[:, -1:]  # last accepted token
            actual_draft_len = min(draft_len, max_new_tokens - tokens_generated)

            for _ in range(actual_draft_len):
                draft_logits, draft_cache, exit_hidden = self._early_exit_forward(
                    input_ids=current_draft_input,
                    exit_layer=assistant_early_exit,
                    past_key_values=draft_cache,
                    return_exit_hidden=True,
                )
                total_generated_tokens += 1
                draft_next = draft_logits[:, -1:].argmax(dim=-1)
                draft_tokens.append(draft_next)
                verify_input_exit_hiddens.append(exit_hidden)
                current_draft_input = draft_next

            if len(draft_tokens) == 0:
                break

            draft_sequence = torch.cat(draft_tokens, dim=-1)  # (1, draft_len)

            # One extra early-exit step on the last drafted token so verifier gets
            # logits for both draft verification and correction token.
            _, draft_cache, exit_hidden_last_draft = self._early_exit_forward(
                input_ids=current_draft_input,
                exit_layer=assistant_early_exit,
                past_key_values=draft_cache,
                return_exit_hidden=True,
            )
            verify_input_exit_hiddens.append(exit_hidden_last_draft)
            verify_input_exit = torch.cat(verify_input_exit_hiddens, dim=1)
            total_drafting_time_s += time.perf_counter() - draft_start_t

            # --- VERIFY PHASE: run only remaining layers on cached exit hidden states ---
            verify_cache = DynamicCache.from_legacy_cache(
                [
                    (full_cache.key_cache[i].clone(), full_cache.value_cache[i].clone())
                    for i in range(len(full_cache))
                ]
            ) if len(full_cache) > 0 else DynamicCache()
            verify_logits, verify_cache = self._remaining_layers_forward_from_exit(
                exit_hidden_states=verify_input_exit,
                exit_layer=assistant_early_exit,
                past_key_values=verify_cache,
            )

            # Compare: verify_logits[:, i] predicts token at position i+1
            # draft_sequence[:, i] is the drafted token at position i
            # verify_logits[:, 0] verifies draft_sequence[:, 0]
            # verify_logits[:, i] verifies draft_sequence[:, i]
            verify_preds = verify_logits[:, :-1].argmax(dim=-1)  # (1, draft_len)

            if tokenizer is not None:
                print(f"Draft sequence: {tokenizer.batch_decode(draft_sequence)}")
                print(f"Verify preds: {tokenizer.batch_decode(verify_preds)}")

            # Find how many consecutive draft tokens match
            matches = (draft_sequence == verify_preds)
            # Find first mismatch
            if matches.all():
                n_accepted = draft_sequence.shape[1]
            else:
                # cumprod finds the longest prefix of matches
                n_accepted = matches.cumprod(dim=1).sum(dim=1).min().item()

            # Record acceptance metrics
            total_accepted_tokens += n_accepted
            total_accepted_lengths.append(n_accepted)
            accept_counts += 1
            for pos in range(actual_draft_len):
                if pos >= len(draft_position_attempt_counts):
                    draft_position_attempt_counts.append(0)
                    draft_position_accept_counts.append(0)
                draft_position_attempt_counts[pos] += 1
                if pos < n_accepted:
                    draft_position_accept_counts[pos] += 1

            # Accept the matching tokens
            if n_accepted > 0:
                generated = torch.cat(
                    [generated, draft_sequence[:, :n_accepted]], dim=-1
                )

            # Add the correction token (verifier's prediction at the mismatch point)
            correction_token = verify_logits[:, n_accepted:n_accepted + 1].argmax(dim=-1)
            generated = torch.cat([generated, correction_token], dim=-1)

            new_tokens = n_accepted + 1
            tokens_generated += new_tokens
            pbar.update(new_tokens)

            # Update caches by truncating to accepted-prefix + correction length.
            target_cache_len = full_cache.get_seq_length() + n_accepted + 1

            # First E layers from draft cache
            for layer_idx in range(min(assistant_early_exit, len(draft_cache))):
                draft_cache.key_cache[layer_idx] = draft_cache.key_cache[layer_idx][
                    ..., :target_cache_len, :
                ]
                draft_cache.value_cache[layer_idx] = draft_cache.value_cache[layer_idx][
                    ..., :target_cache_len, :
                ]

            # Remaining layers from verify cache
            for layer_idx in range(len(verify_cache)):
                if layer_idx < assistant_early_exit:
                    continue
                verify_cache.key_cache[layer_idx] = verify_cache.key_cache[layer_idx][
                    ..., :target_cache_len, :
                ]
                verify_cache.value_cache[layer_idx] = verify_cache.value_cache[layer_idx][
                    ..., :target_cache_len, :
                ]

            # Merge layer caches back into full cache
            full_cache = verify_cache
            for layer_idx in range(min(assistant_early_exit, len(full_cache), len(draft_cache))):
                full_cache.key_cache[layer_idx] = draft_cache.key_cache[layer_idx]
                full_cache.value_cache[layer_idx] = draft_cache.value_cache[layer_idx]

            # Check stopping criteria
            if stopping_criteria is not None:
                is_done = stopping_criteria(
                    input_ids=generated[:, prompt_len:],
                    scores=None,
                )
                if torch.any(is_done):
                    done = True

        pbar.close()

        if tokenizer is not None:
            acceptance_rate = (
                total_accepted_tokens / total_generated_tokens
                if total_generated_tokens > 0
                else 0
            )
            avg_accepted_len = (
                sum(total_accepted_lengths) / accept_counts
                if accept_counts > 0
                else 0
            )
            print(f"[LayerSkip] Total drafted tokens: {total_generated_tokens}")
            print(f"[LayerSkip] Total accepted tokens: {total_accepted_tokens}")
            print(f"[LayerSkip] Acceptance rate: {acceptance_rate:.2%}")
            print(f"[LayerSkip] Avg accepted length: {avg_accepted_len:.2f}")
            print(tokenizer.batch_decode(generated))

        self._last_draft_position_acceptance = {
            "attempt_counts": draft_position_attempt_counts,
            "accept_counts": draft_position_accept_counts,
            "acceptance_rates": [
                (acc / att) if att > 0 else 0.0
                for acc, att in zip(draft_position_accept_counts, draft_position_attempt_counts)
            ],
        }

        total_all_time_s = time.perf_counter() - overall_start_t
        return (
            generated,
            (total_generated_tokens, total_accepted_tokens),
            (total_accepted_lengths, accept_counts),
            (total_drafting_time_s, total_all_time_s),
        )
