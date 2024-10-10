import logging
import os
from typing import Any, Dict, Optional, Set, Tuple, Union

import huggingface_hub
import peft
import torch
import torch.nn as nn
import transformers
import transformers.activations
import transformers.modeling_outputs
import transformers.models

import ultravox.model.ultravox_model as ultravox_model

# from third_party.tokenizer import wav_tokenizer
from third_party.tokenizer.decoder import pretrained as wav_tokenizer
from ultravox.model import ultravox_config
from ultravox.ultravoxls import ultravoxls_config


class UltravoxLSModel(transformers.LlamaPreTrainedModel):
    """Ultravox Language-Speech Model. A pretrained llama backbone trained on speech tokens."""

    config_class = ultravoxls_config.UltravoxLSConfig
    config: ultravoxls_config.UltravoxLSConfig  # for type hinting
    # We don't store the tokenizer in the state_dict since it is not trained
    _keys_to_ignore_on_load_missing = ["tokenizer.*"]
    # We minimize the weights in state_dict in order to reduce the size of the checkpoint
    # The issue is that load_pretrained() uses state_dict() keys to know what keys are expected
    # As such we have to tell is to ignore some keys that are not always in the model
    _keys_to_ignore_on_load_unexpected = ["language_model.*"]
    # Usually we load encoder weights from a pretrained model, so we don't want to load the decoder weights
    # Technically we never hit this issue because these keys are already removed from state_dict() however,
    # but there's no harm in keeping it here for when we change that behavior.

    def __init__(self, config: ultravoxls_config.UltravoxLSConfig):
        super().__init__(config)
        self._register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)

        self.keep_params: Set[str] = set()
        self.vocab_size = config.vocab_size

        self.tokenizer = self._create_tokenizer(config)
        self.tokenizer.eval()

        self.language_model = ultravox_model.UltravoxModel._create_language_model(
            config
        )
        self.resize_token_embeddings(self.tokenizer_vocab_size)

        # Arbitrary value for the pad token id. The attention mask will be set to 0 for these tokens in the data collator.
        self.pad_token_id = 0

        self.loss_config = ultravox_config.LossConfig()
        self.post_init()

    def train(self, mode: bool = True) -> "UltravoxLSModel":
        super().train(mode)
        # Tokenizer must always be in eval mode
        self.tokenizer.eval()
        return self

    def init_weights(self):
        super().init_weights()

        # We need to explicitly re-initialize embeddings, o.w. the text embeddings are used
        if transformers.modeling_utils._init_weights:
            self._init_weights(self.get_input_embeddings())

    def _create_tokenizer(
        self, config: ultravoxls_config.UltravoxLSConfig
    ) -> wav_tokenizer.WavTokenizer:
        HF_MODEL_NAME = "novateur/WavTokenizer"
        CHECKPOINT_FILE_NAME = "WavTokenizer_small_600_24k_4096.ckpt"

        model_path = huggingface_hub.hf_hub_download(
            repo_id=HF_MODEL_NAME, filename=CHECKPOINT_FILE_NAME
        )

        # TODO: this file should work outside the context of the Ultravox repo as well, hence the following line should be adjusted
        config_path = "third_party/tokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        config_path = os.path.join(root_dir, config_path)

        tokenizer: wav_tokenizer.WavTokenizer = (
            wav_tokenizer.WavTokenizer.from_pretrained0802(config_path, model_path)
        )

        # freeze the tokenizer
        for param in tokenizer.parameters():
            param.requires_grad = False

        tokenizer.apply(lambda module: setattr(module, "_is_hf_initialized", True))

        # we don't convert tokenizer to dtype here since tokenizer and LLM can have different dtypes

        return tokenizer

    def _convert_tokens_to_wav(self, discrete_code):
        features = self.tokenizer.codes_to_features(discrete_code)
        bandwidth_id = torch.tensor([0])
        audio_out = self.tokenizer.decode(features, bandwidth_id=bandwidth_id)
        return audio_out.cpu()

    @property
    def tokenizer_vocab_size(self):
        return self.tokenizer.feature_extractor.encodec.quantizer.bins

    def _convert_wav_to_tokens(self, wav: torch.Tensor):
        wav = wav.to(self.device)
        bandwidth_id = torch.tensor([0], device=wav.device)
        _, discrete_code = self.tokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)

        return discrete_code

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def set_loss_config(self, loss_config: ultravox_config.LossConfig):
        self.loss_config = loss_config

    def _setup_cache(
        self, cache_cls, max_batch_size: int, max_cache_len: Optional[int] = None
    ):
        self.language_model._setup_cache(cache_cls, max_batch_size, max_cache_len)

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.language_model._reorder_cache(past_key_values, beam_idx)

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of
        )
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _compute_tokens_from_audio(self, audio: torch.Tensor, num_tokens: torch.Tensor):
        with torch.no_grad():
            # clone is needed since tokenizer uses inference mode which is not compatible with autograd
            input_ids = self._convert_wav_to_tokens(audio).squeeze(0).clone()
            pad_to = self.config.pad_to_multiple_of
            max_num_tokens = (num_tokens.max() + pad_to - 1) // pad_to * pad_to
            input_ids = nn.functional.pad(
                input_ids,
                (0, max_num_tokens - input_ids.shape[-1]),
                value=self.pad_token_id,
            )
            # generate attention mask using num_tokens, e.g. 1 1 1 1 0 0 ... if num_tokens is 4
            attention_mask = torch.stack(
                [
                    torch.arange(max_num_tokens, device=audio.device) < num_tokens_i
                    for num_tokens_i in num_tokens
                ]
            ).long()
            labels = torch.where(attention_mask.bool(), input_ids, -100)

        return input_ids, attention_mask, labels

    def forward(
        self,
        audio: torch.Tensor,
        num_tokens: torch.Tensor,
        return_loss: bool = True,
        **kwargs,
    ) -> Union[Tuple, transformers.modeling_outputs.CausalLMOutputWithPast]:
        """
        Forward pass for the UltravoxLS model.

        Args:
            audio: The audio input.
            num_tokens: The number of tokens expected to be produced from the audio input. Used to generate attention mask.
                Since the original audio is padded, this value is used to let us ignore the padded tokens.
                If the audio is padded, the last token is also ignored as it is not fully valid (i.e. affected by padding).
            **kwargs: Additional keyword arguments. Passed directly to the language model.
        """
        input_ids, attention_mask, labels = self._compute_tokens_from_audio(
            audio, num_tokens
        )

        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def generate(self, audio: torch.Tensor, num_tokens: torch.Tensor, **kwargs):
        assert (
            audio.shape[0] == 1
        ), f"Generate with larger batch size is not supported, got audio with shape {audio.shape}"
        input_len = audio.shape[-1]

        input_ids, attention_mask, _ = self._compute_tokens_from_audio(
            audio, num_tokens
        )

        output = self.language_model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        output_tokens = output.sequences[0]
        output_tokens = output_tokens.reshape(1, 1, -1)
        output_audio = self._convert_tokens_to_wav(output_tokens)

        output_tokens_len = output_tokens.shape[-1] - num_tokens[0]

        return output_audio[..., input_len:], output_tokens_len

    def merge_and_unload(self):
        if isinstance(self.language_model, peft.PeftModel):
            self.language_model = self.language_model.merge_and_unload()
            # no need to download base language model weights anymore, so we can remove the id
            self.config.text_model_id = None
            self.keep_params.update(
                set(
                    [
                        f"language_model.{name}"
                        for name, _ in self.language_model.named_parameters()
                    ]
                )
            )

        for param in ["text_model_lora_config"]:
            if hasattr(self.config, param):
                delattr(self.config, param)

    def push_to_hub(self, *args, **kwargs):
        self.merge_and_unload()
        return super().push_to_hub(*args, **kwargs)

    def save_pretrained(
        self, *args, state_dict: Optional[Dict[str, Any]] = None, **kwargs
    ):
        if state_dict is None:
            state_dict = super().state_dict()

        named_params = dict(self.named_parameters())

        state_dict = {
            k: v
            for k, v in state_dict.items()
            if k in self.keep_params
            or (k in named_params and named_params[k].requires_grad)
        }

        super().save_pretrained(*args, state_dict=state_dict, **kwargs)

    def _pre_load_state_dict_hook(self, state_dict: Dict[str, Any], *args, **kwargs):
        self.keep_params.update(set(state_dict.keys()))

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model (reuses Peft model's method)
        """
        count_params = peft.peft_model.PeftModel.get_nb_trainable_parameters

        trainable_params, all_param = count_params(self.language_model)

        logging.info(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d}"
            f" || trainable%: {100 * trainable_params / all_param:.1f}%"
        )
