# -----------------------------------------------------------------------
# S2Tab official code : models/generation_utils_multi_task_mult_ver2_empty.py
# -----------------------------------------------------------------------
# Modified from huggingface (https://github.com/huggingface/transformers/generation_utils.py)
# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------

from typing import Optional, Union, Iterable, List, Callable, Tuple, Dict, TYPE_CHECKING
import warnings
import inspect
from dataclasses import dataclass
import math
import copy

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers import GenerationMixin
from transformers.utils import logging
from transformers.generation.beam_constraints import Constraint, DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.utils import ModelOutput
from transformers.generation.configuration_utils import GenerationConfig

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)

class LogitsProcessorListMultiTask(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] or [`LogitsWarper`] to subsequently process a
    `scores` input tensor. This class inherits from list and adds a specific *__call__* method to apply each
    [`LogitsProcessor`] or [`LogitsWarper`] to the inputs.
    """

    def __call__(self, input_ids , scores, **kwargs):
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters # self, input_ids, scores
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)
        return scores

def _get_decoder_start_token_id(self, decoder_start_token_id_row: int = None, 
                                      decoder_start_token_id_col: int = None,
                                      decoder_start_coord_ids: List[List[int]] = None, 
                                      decoder_start_delta_vars: List[List[float]] = None, 
                                      bos_token_id: int = None) -> Tuple[float, int]:
    decoder_start_token_id_row = (
        decoder_start_token_id_row 
        if decoder_start_token_id_row is not None 
        else self.config.decoder_start_token_id
    ) # token_type
    decoder_start_token_id_col = (
        decoder_start_token_id_col 
        if decoder_start_token_id_col is not None 
        else self.config.decoder_start_token_id
    ) # token_type

    if decoder_start_token_id_row is not None:
        pass
    elif bos_token_id is not None:
        decoder_start_token_id_row = bos_token_id
    else:
        raise ValueError(
            "`decoder_start_token_id_row` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    if decoder_start_token_id_col is not None:
        pass
    elif bos_token_id is not None:
        decoder_start_token_id_col = bos_token_id
    else:
        raise ValueError(
            "`decoder_start_token_id_col` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    decoder_start_coord_ids = (
        decoder_start_coord_ids if decoder_start_coord_ids is not None else [[[[bos_token_id, bos_token_id]] * 8]]
    )

    decoder_start_delta_vars = (
        decoder_start_delta_vars if decoder_start_delta_vars is not None else [[[0., 1.] * 4]]
    )

    return {"coord": decoder_start_coord_ids, 
            "delta": decoder_start_delta_vars,
            "token_row": decoder_start_token_id_row, 
            "token_col": decoder_start_token_id_col, 
            }

    
def _prepare_decoder_input_ids_for_generation(
    self,
    batch_size: int,
    model_input_name: str,
    model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    decoder_start_token_id_row: int = None,
    decoder_start_token_id_col: int = None,
    decoder_start_coord_ids: List[List[int]] = None,
    decoder_start_delta_vars: float = None,
    bos_token_id: int = None,
    device: torch.device = None,
) -> Tuple[torch.FloatTensor, torch.LongTensor]:    
    """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
    # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
    # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
    if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
        decoder_input_ids = model_kwargs.pop("decoder_input_ids")
    elif "input_ids" in model_kwargs and model_input_name != "input_ids":
        decoder_input_ids = model_kwargs.pop("input_ids")
    else:
        decoder_input_ids = None

    
    # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
    if decoder_start_token_id_row is None and bos_token_id is not None:
        decoder_start_token_id_row = bos_token_id
        decoder_start_token_id_col = bos_token_id
    elif decoder_start_token_id_row is None and bos_token_id is None:
        raise ValueError("'decoder_start_token_id' or 'bos_token_id' are required.")
    decoder_start_token_id_dict = self._get_decoder_start_token_id(decoder_start_token_id_row, 
                                                                   decoder_start_token_id_col,
                                                                   decoder_start_coord_ids, 
                                                                   decoder_start_delta_vars, 
                                                                   bos_token_id)
    if device is None:
        device = self.device
    decoder_start_coord_ids = decoder_start_token_id_dict['coord']
    if len(decoder_start_coord_ids) == 1 and batch_size > 1:
        decoder_start_coord_ids = [decoder_start_coord_ids[0]] * batch_size # bs, 1, 8, 2
    decoder_start_coord_ids = torch.as_tensor(decoder_start_coord_ids).to(device)
    decoder_start_delta_vars = decoder_start_token_id_dict['delta']
    if len(decoder_start_delta_vars) == 1 and batch_size > 1:
        decoder_start_delta_vars = [decoder_start_delta_vars[0]] * batch_size
    decoder_start_delta_vars = torch.as_tensor(decoder_start_delta_vars).to(device)
    
    decoder_start_token_id_row = decoder_start_token_id_dict['token_row']
    decoder_start_token_id_row = torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id_row
    decoder_start_token_id_col = decoder_start_token_id_dict['token_col']
    decoder_start_token_id_col = torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id_col
    
    # no user input -> use decoder_start_token_id as decoder_input_ids
    if decoder_input_ids is None:
        decoder_input_ids = [decoder_start_coord_ids, decoder_start_token_id_row, decoder_start_token_id_col, decoder_start_delta_vars]
    # exception: Donut checkpoints have task-specific decoder starts and don't expect a BOS token
    elif self.config.model_type == 'vision-encoder-decoder' and 'donut' in self.name_or_path.lower():
        pass
    # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
    # decoder_attention_mask if provided)
    # NOT IMPLEMENTED. JUST COPIED from transformers/generation/utils.py line 675-683
    elif (decoder_input_ids[:, 0] != decoder_start_token_id_row).all().item(): 
        decoder_input_ids = torch.cat([decoder_start_token_id_dict['token'], decoder_input_ids], dim=-1)
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            decoder_attention_mask = torch.cat(
                (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                dim=-1,
            )
            model_kwargs["decoder_attention_mask"] = decoder_attention_mask


    return decoder_input_ids, model_kwargs



def _get_logits_processor(
    self,
    generation_config: GenerationConfig,
    input_ids_seq_length: int,
    encoder_input_ids: torch.LongTensor,
    prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
    logits_processor: Optional[LogitsProcessorListMultiTask],
) -> LogitsProcessorListMultiTask:
    """
    This class returns a [`LogitsProcessorListMultiTask`] list object that contains all relevant [`LogitsProcessor`]
    instances used to modify the scores of the language model head.
    """
    processors = LogitsProcessorListMultiTask()

    # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
    # all samplers can be found in `generation_utils_samplers.py`
    if generation_config.diversity_penalty is not None and generation_config.diversity_penalty > 0.0:
        processors.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=generation_config.diversity_penalty,
                num_beams=generation_config.num_beams,
                num_beam_groups=generation_config.num_beam_groups,
            )
        )
    if (
        generation_config.encoder_repetition_penalty is not None
        and generation_config.encoder_repetition_penalty != 1.0
    ):
        processors.append(
            EncoderRepetitionPenaltyLogitsProcessor(
                penalty=generation_config.encoder_repetition_penalty, encoder_input_ids=encoder_input_ids
            )
        )
    if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))
    if generation_config.no_repeat_ngram_size is not None and generation_config.no_repeat_ngram_size > 0:
        processors.append(NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))
    if (
        generation_config.encoder_no_repeat_ngram_size is not None
        and generation_config.encoder_no_repeat_ngram_size > 0
    ):
        if self.config.is_encoder_decoder:
            processors.append(
                EncoderNoRepeatNGramLogitsProcessor(
                    generation_config.encoder_no_repeat_ngram_size, encoder_input_ids
                )
            )
        else:
            raise ValueError(
                "It's impossible to use `encoder_no_repeat_ngram_size` with decoder-only architecture"
            )
    if generation_config.bad_words_ids is not None:
        processors.append(
            NoBadWordsLogitsProcessor(generation_config.bad_words_ids, generation_config.eos_token_id)
        )
    if (
        generation_config.min_length is not None
        and generation_config.eos_token_id is not None
        and generation_config.min_length > 0
    ):
        processors.append(MinLengthLogitsProcessor(generation_config.min_length, generation_config.eos_token_id))
    if (
        generation_config.min_new_tokens is not None
        and generation_config.eos_token_id is not None
        and generation_config.min_new_tokens > 0
    ):
        processors.append(
            MinNewTokensLengthLogitsProcessor(
                input_ids_seq_length, generation_config.min_new_tokens, generation_config.eos_token_id
            )
        )
    if prefix_allowed_tokens_fn is not None:
        processors.append(
            PrefixConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn, generation_config.num_beams // generation_config.num_beam_groups
            )
        )
    if generation_config.forced_bos_token_id is not None:
        processors.append(ForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id))
    if generation_config.forced_eos_token_id is not None:
        processors.append(
            ForcedEOSTokenLogitsProcessor(generation_config.max_length, generation_config.forced_eos_token_id)
        )
    if generation_config.remove_invalid_values is True:
        processors.append(InfNanRemoveLogitsProcessor())
    if generation_config.exponential_decay_length_penalty is not None:
        processors.append(
            ExponentialDecayLengthPenalty(
                generation_config.exponential_decay_length_penalty,
                generation_config.eos_token_id,
                input_ids_seq_length,
            )
        )
    if generation_config.suppress_tokens is not None:
        processors.append(SuppressTokensLogitsProcessor(generation_config.suppress_tokens))
    if generation_config.begin_suppress_tokens is not None:
        begin_index = input_ids_seq_length
        begin_index = (
            begin_index
            if (input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None)
            else begin_index + 1
        )
        if generation_config.forced_decoder_ids is not None:
            # generation starts after the last token that is forced
            begin_index += generation_config.forced_decoder_ids[-1][0]
        processors.append(
            SuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, begin_index)
        )
    if generation_config.forced_decoder_ids is not None:
        processors.append(ForceTokensLogitsProcessor(generation_config.forced_decoder_ids))
    processors = self._merge_criteria_processor_list(processors, logits_processor)
    # `LogitNormalization` should always be the last logit processor, when present
    if generation_config.renormalize_logits is True:
        processors.append(LogitNormalization())
    return processors

@torch.no_grad()
def generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorListMultiTask] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    **kwargs,
    ): #-> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
    r"""

    Generates sequences of token ids for models with a language modeling head.

    <Tip warning={true}>

    Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
    model's default generation configuration. You can override any `generation_config` by passing the corresponding
    parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

    For an overview of generation strategies and code examples, check out the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
            should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
            `input_ids`, `input_values`, `input_features`, or `pixel_values`.
        generation_config (`~generation.GenerationConfig`, *optional*):
            The generation configuration to be used as base parametrization for the generation call. `**kwargs`
            passed to generate matching the attributes of `generation_config` will override them. If
            `generation_config` is not provided, the default will be used, which had the following loading
            priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
            configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
            default values, whose documentation should be checked to parameterize generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            Custom logits processors that complement the default logits processors built from arguments and
            generation config. If a logit processor is passed that is already created with the arguments or a
            generation config an error is thrown. This feature is intended for advanced users.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            Custom stopping criteria that complement the default stopping criteria built from arguments and a
            generation config. If a stopping criteria is passed that is already created with the arguments or a
            generation config an error is thrown. This feature is intended for advanced users.
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
            If provided, this function constraints the beam search to allowed tokens only at each step. If not
            provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
            `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
            on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
            for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
            Retrieval](https://arxiv.org/abs/2010.00904).
        synced_gpus (`bool`, *optional*):
            Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
            `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
            generating before other GPUs. Otherwise it'll be set to `False`.
        assistant_model (`PreTrainedModel`, *optional*):
            An assistant model that can be used to accelerate generation. The assistant model must have the exact
            same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
            is much faster than running generation with the model you're calling generate from. As such, the
            assistant model should be much smaller.
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        kwargs:
            Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
            forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
            specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

    Return:
        [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
        or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

            If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GreedySearchDecoderOnlyOutput`],
                - [`~generation.SampleDecoderOnlyOutput`],
                - [`~generation.BeamSearchDecoderOnlyOutput`],
                - [`~generation.BeamSampleDecoderOnlyOutput`]

            If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GreedySearchEncoderDecoderOutput`],
                - [`~generation.SampleEncoderDecoderOutput`],
                - [`~generation.BeamSearchEncoderDecoderOutput`],
                - [`~generation.BeamSampleEncoderDecoderOutput`]
    ```"""
    if synced_gpus is None:
        if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
            synced_gpus = True
        else:
            synced_gpus = False

    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    self._validate_model_class()

    # priority: `generation_config` argument > `model.generation_config` (the default generation config)
    if generation_config is None:
        # legacy: users may modify the model configuration to control generation -- update the generation config
        # model attribute accordingly, if it was created from the model config
        if self.generation_config._from_model_config:
            new_generation_config = GenerationConfig.from_model_config(self.config)
            # manually add decoder_start_token_id for row / col, respectively
            setattr(new_generation_config, "decoder_start_token_id_row", self.config.decoder_start_token_id_row)
            setattr(new_generation_config, "decoder_start_token_id_col", self.config.decoder_start_token_id_col)
            if new_generation_config != self.generation_config:
                warnings.warn(
                    "You have modified the pretrained model configuration to control generation. This is a"
                    " deprecated strategy to control generation and will be removed soon, in a future version."
                    " Please use a generation configuration file (see"
                    " https://huggingface.co/docs/transformers/main_classes/text_generation)"
                )
                self.generation_config = new_generation_config
        generation_config = self.generation_config

    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
    generation_config.validate()
    self._validate_model_kwargs(model_kwargs.copy())

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
        if model_kwargs.get("attention_mask", None) is None:
            logger.warning(
                "The attention mask and the pad token id were not set. As a consequence, you may observe "
                "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
            )
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        generation_config.pad_token_id = eos_token_id

    # 3. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    # 4. Define other model kwargs
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    model_kwargs["use_cache"] = generation_config.use_cache

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )

    # decoder-only models should use left-padding for generation
    if not self.config.is_encoder_decoder:
        if (
            generation_config.pad_token_id is not None
            and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    # encoder_table_box = self.box_regression(model_kwargs['encoder_outputs'].pooler_output[:, None, :]) # bs,1,4
    # encoder_table_box = torch.sigmoid(encoder_table_box)
    # # decoder_start_coord = torch.zeros((batch_size, 1, 4)).to(inputs.device)
    
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id_row=generation_config.decoder_start_token_id_row,
            decoder_start_token_id_col=generation_config.decoder_start_token_id_col,
            decoder_start_coord_ids=[[[[generation_config.bos_token_id, generation_config.bos_token_id]] * 8]], # [bs,1,8,2]
            decoder_start_delta_vars=[[[0., 1.] * 4]], # [bs,1,8]
            bos_token_id=generation_config.bos_token_id,
            device=inputs_tensor.device,
        ) # (coord, token, endrow)
    else:
        # if decoder-only then inputs_tensor has to be `input_ids`
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_seq_length = input_ids[1].shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
            "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
            " recommend using `max_new_tokens` to control the maximum length of the generation.",
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        if not has_default_max_length:
            logger.warning(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )
        generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

    if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
        raise ValueError(
            f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
            f" the maximum length ({generation_config.max_length})"
        )
    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
            f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
            " increasing `max_new_tokens`."
        )

    # 7. determine generation mode
    is_constraint_gen_mode = (
        generation_config.constraints is not None or generation_config.force_words_ids is not None
    )

    is_contrastive_search_gen_mode = (
        (generation_config.num_beams == 1)
        and generation_config.top_k is not None
        and generation_config.top_k > 1
        and generation_config.do_sample is False
        and generation_config.penalty_alpha is not None
        and generation_config.penalty_alpha > 0
    )

    is_greedy_gen_mode = (
        (generation_config.num_beams == 1)
        and (generation_config.num_beam_groups == 1)
        and generation_config.do_sample is False
        and not is_constraint_gen_mode
        and not is_contrastive_search_gen_mode
    )
    is_sample_gen_mode = (
        (generation_config.num_beams == 1)
        and (generation_config.num_beam_groups == 1)
        and generation_config.do_sample is True
        and not is_constraint_gen_mode
        and not is_contrastive_search_gen_mode
    )
    is_beam_gen_mode = (
        (generation_config.num_beams > 1)
        and (generation_config.num_beam_groups == 1)
        and generation_config.do_sample is False
        and not is_constraint_gen_mode
        and not is_contrastive_search_gen_mode
    )
    is_beam_sample_gen_mode = (
        (generation_config.num_beams > 1)
        and (generation_config.num_beam_groups == 1)
        and generation_config.do_sample is True
        and not is_constraint_gen_mode
        and not is_contrastive_search_gen_mode
    )
    is_group_beam_gen_mode = (
        (generation_config.num_beams > 1)
        and (generation_config.num_beam_groups > 1)
        and not is_constraint_gen_mode
        and not is_contrastive_search_gen_mode
    )
    is_assisted_gen_mode = False
    if assistant_model is not None:
        if not (is_greedy_gen_mode or is_sample_gen_mode):
            raise ValueError(
                "You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate "
                "is only supported with Greedy Search and Sample."
            )
        is_assisted_gen_mode = True

    if generation_config.num_beam_groups > generation_config.num_beams:
        raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
    if is_group_beam_gen_mode and generation_config.do_sample is True:
        raise ValueError(
            "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
        )

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if self.device.type != input_ids[1].device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 8. prepare distribution pre_processing samplers
    logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    # 9. prepare stopping criteria
    stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )

    # 10. go into different generation modes - only greedy_gen_mode is valid!
    if is_greedy_gen_mode:
        if generation_config.num_return_sequences > 1:
            raise ValueError(
                "num_return_sequences has to be 1 when doing greedy search, "
                f"but is {generation_config.num_return_sequences}."
            )

        # 10. run greedy search
        return self.greedy_search(
            input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif is_sample_gen_mode:
        # 10. prepare logits warper
        logits_warper = self._get_logits_warper(generation_config)

        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 12. run sample
        return self.sample(
            input_ids,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )


@dataclass
class GreedySearchEncoderDecoderOutputMultiTask(ModelOutput):
    sequences: torch.FloatTensor = None
    scores: Optional[Dict[str, Tuple[torch.FloatTensor]]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    encoder_pooler_output: torch.FloatTensor = None
    entire_raw_scores: torch.FloatTensor = None
    encoder_feature_size: Tuple = None

def greedy_search(
    self,
    input_ids: Tuple[torch.FloatTensor, torch.LongTensor],
    logits_processor: Optional[LogitsProcessorListMultiTask] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: Optional[bool] = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GreedySearchEncoderDecoderOutputMultiTask, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
    used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorListMultiTask`, *optional*):
            An instance of [`LogitsProcessorListMultiTask`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.

        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
            If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation_utils.GreedySearchDecoderOnlyOutput`], [`~generation_utils.GreedySearchEncoderDecoderOutput`]
        or `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation_utils.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation_utils.GreedySearchEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    ```"""
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorListMultiTask()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    output_scores = output_scores if output_scores is not None else self.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores_row = () if (return_dict_in_generate and output_scores) else None
    scores_col = () if (return_dict_in_generate and output_scores) else None
    empty_scores = () if (return_dict_in_generate and output_scores) else None
    coord_logits = () if (return_dict_in_generate and output_scores) else None
    ocr_logits = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )
    

    # keep track of which sequences are already finished
    unfinished_sequences = input_ids[1].new(input_ids[1].shape[0]).fill_(1)
    cur_len = input_ids[1].shape[-1]

    this_peer_finished = False  # used by synced_gpus only
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_coord_logits = outputs.continuous_logits[:, -1, :] # bs, 4
        next_token_logits_row = outputs.logits[:, -1, :] # bs, class
        next_token_logits_col = outputs.logits_col[:, -1, :] # bs, class
        if hasattr(outputs, "empty_logits"):
            next_empty_logits = torch.sigmoid(outputs.empty_logits[:, -1, :])
        else:
            next_empty_logits = None
        if hasattr(outputs, "vision_decoder_logits") and outputs.vision_decoder_logits is not None:
            next_ocr_logits = outputs.vision_decoder_logits[:, -1, :, :] # (B, max_token_length, class)
        else:
            next_ocr_logits = None

        # pre-process distribution
        # next_tokens_scores 
        next_coords_ids, next_delta_vars, next_tokens_scores_row, next_tokens_scores_col, next_coords = \
                    logits_processor(input_ids, tuple([next_token_coord_logits, next_token_logits_row, next_token_logits_col]))

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores_row += tuple([next_tokens_scores_row])
                scores_col += tuple([next_tokens_scores_col])
                coord_logits += tuple([next_token_coord_logits])
                if next_empty_logits is not None:
                    empty_scores += tuple([next_empty_logits])
                if next_ocr_logits is not None:
                    ocr_logits += tuple([next_ocr_logits])

            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # argmax
        next_tokens_row = torch.argmax(next_tokens_scores_row, dim=-1)
        next_tokens_col = torch.argmax(next_tokens_scores_col, dim=-1)
        sep_mask = torch.logical_or(next_tokens_row == self.config.sep_token_id, next_tokens_col == self.config.sep_token_id)
        if self.config.eoh_token_id is not None:
            eoh_mask = torch.logical_or(next_tokens_row == self.config.eoh_token_id, next_tokens_col == self.config.eoh_token_id)
        sep_mask = sep_mask.unsqueeze(1).unsqueeze(2).repeat((1,next_coords_ids.shape[1], next_coords_ids.shape[2]))
        if self.config.eoh_token_id is not None:
            eoh_mask = eoh_mask.unsqueeze(1).unsqueeze(2).repeat((1,next_coords_ids.shape[1], next_coords_ids.shape[2]))
        next_coords_ids = torch.where(sep_mask == True, torch.zeros_like(next_coords_ids) + self.config.sep_token_id, next_coords_ids) # change coord ids to SEP
        if self.config.eoh_token_id is not None:
            next_coords_ids = torch.where(eoh_mask == True, torch.zeros_like(next_coords_ids) + self.config.eoh_token_id, next_coords_ids)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens_row = next_tokens_row * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            next_tokens_col = next_tokens_col * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = update_input_ids(input_ids, (next_coords_ids, next_tokens_row, next_tokens_col, next_delta_vars))
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        cur_len = cur_len + 1

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens_row != eos_token_id).long())

        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids[1], scores_row):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GreedySearchEncoderDecoderOutputMultiTask(
                sequences=input_ids,
                scores={"type_scores_row": scores_row,
                        "type_scores_col": scores_col,
                        "empty_scores": empty_scores,
                        "coord_logits": coord_logits,
                        "ocr_logits": ocr_logits,
                    },
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                entire_raw_scores=outputs.logits,
                encoder_feature_size=outputs.encoder_feature_size,
            )
        else:
            raise ValueError("only decoder option is not implemented.")
    else:
        return input_ids


def update_input_ids(input_id: Tuple[torch.FloatTensor, torch.LongTensor], 
                    next_token_id: Tuple[torch.FloatTensor, torch.LongTensor]):
    updated_input_id = []
    for prev, next in zip(input_id, next_token_id):
        if next is None:
            updated = None
        elif len(next.shape) == 2:
            updated = torch.cat((prev, next[:, None, :]), dim=1)
        elif len(next.shape) == 1:
            updated = torch.cat((prev, next[:, None]), dim=1)
        elif len(next.shape) == 3:
            updated = torch.cat((prev, next[:, None, :, :]), dim=1)
        else:
            raise ValueError(f"update rule for tensor of shape {next.shape} has not been implemented.")
        updated_input_id.append(updated)
    
    return tuple(updated_input_id)

class TableLogitProcessor():
    def __init__(self, type_vocab_size=5, coord_bin=400, coord_offset=5):
        self.type_vocab_size = type_vocab_size
        self.coord_bin = coord_bin
        self.coord_offset = coord_offset

    def __call__(self, input_ids: Tuple[torch.FloatTensor, torch.LongTensor], scores: Tuple[torch.FloatTensor]) -> Tuple[torch.FloatTensor]:
        """Torch method for processing table sequence logits."""
        # Calculate next-token prob
        bs = input_ids[0].shape[0]
        coord_logit = scores[0] #scores[..., :mixture_len] # (pi, mu-4, gamma-4)
        token_type_logit_row = scores[1] #scores[..., mixture_len:mixture_len+token_type_len]
        token_type_logit_col = scores[2]

        # coordinate - first, calculate the coordinates!
        next_coords = torch.sigmoid(coord_logit)
        next_coords_xyxy = next_coords.clone()
        next_coords_xyxy[..., 2:] += next_coords_xyxy[..., :2]
        next_coords_xyxy.clamp_(max=1.)
        

        next_coord_x1, next_coord_y1, next_coord_x2, next_coord_y2 = next_coords_xyxy.split(1, dim=-1)
        next_coord_1 = torch.cat((next_coord_x1, next_coord_y1), dim=1) * (self.coord_bin - 1)
        next_coord_2 = torch.cat((next_coord_x2, next_coord_y2), dim=1) * (self.coord_bin - 1)

        next_coord_111 = torch.cat((torch.floor(next_coord_1[..., :1]), torch.floor(next_coord_1[..., 1:])), dim=1)
        next_coord_112 = torch.cat((torch.floor(next_coord_1[..., :1]), torch.ceil(next_coord_1[..., 1:])), dim=1)
        next_coord_121 = torch.cat((torch.ceil(next_coord_1[..., :1]), torch.floor(next_coord_1[..., 1:])), dim=1)
        next_coord_122 = torch.cat((torch.ceil(next_coord_1[..., :1]), torch.ceil(next_coord_1[..., 1:])), dim=1)
        next_coord_211 = torch.cat((torch.floor(next_coord_2[..., :1]), torch.floor(next_coord_2[..., 1:])), dim=1)
        next_coord_212 = torch.cat((torch.floor(next_coord_2[..., :1]), torch.ceil(next_coord_2[..., 1:])), dim=1)
        next_coord_221 = torch.cat((torch.ceil(next_coord_2[..., :1]), torch.floor(next_coord_2[..., 1:])), dim=1)
        next_coord_222 = torch.cat((torch.ceil(next_coord_2[..., :1]), torch.ceil(next_coord_2[..., 1:])), dim=1)
        next_coord_ids = torch.stack((next_coord_111, next_coord_112, 
                                      next_coord_121, next_coord_122,
                                      next_coord_211, next_coord_212,
                                      next_coord_221, next_coord_222,),
                                      dim=1) # (bs, 8, 2)

        next_delta_x11 = next_coord_1[..., 0] - next_coord_111[..., 0]
        next_delta_x12 = torch.ones_like(next_delta_x11) - next_delta_x11
        next_delta_y11 = next_coord_1[..., 1] - next_coord_111[..., 1]
        next_delta_y12 = torch.ones_like(next_delta_y11) - next_delta_y11
        next_delta_x21 = next_coord_2[..., 0] - next_coord_211[..., 0]
        next_delta_x22 = torch.ones_like(next_delta_x21) - next_delta_x21
        next_delta_y21 = next_coord_2[..., 1] - next_coord_211[..., 1]
        next_delta_y22 = torch.ones_like(next_delta_y21) - next_delta_y21
        next_delta_vars = torch.stack((next_delta_x11, next_delta_x12, 
                                       next_delta_y11, next_delta_y12,
                                       next_delta_x21, next_delta_x22,
                                       next_delta_y21, next_delta_y22,),
                                       dim=1) # (bs, 8)

        # token_type
        next_tokens_scores_row = F.softmax(token_type_logit_row, 1) # (bs, class_num)
        next_tokens_scores_col = F.softmax(token_type_logit_col, 1) # (bs, class_num)

        return next_coord_ids.to(torch.long) + self.coord_offset, next_delta_vars, \
               next_tokens_scores_row, next_tokens_scores_col, \
               next_coords
        
