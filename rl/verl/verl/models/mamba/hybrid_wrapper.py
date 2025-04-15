# Copyright (c) 2023, Albert Gu, Tri Dao.
import os
import json

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM

from mamba_ssm.utils.hf import load_config_hf
from transformers.utils.hub import cached_file

from verl.models.mamba.hybrid_mamba_config import MambaConfig
from verl.models.mamba.hybrid_model import MambaDecoderLayer, MHADecoderLayer

from verl.models.mamba.util import load_safetensors_to_dict, load_state_dict_hf
from verl.models.mamba._generation import GenerationMixin
from collections import namedtuple

MAMBA_CONFIG_NAME = "mamba_config.json"

class MambaTransformerHybridModelWrapper(nn.Module, GenerationMixin):
    def __init__(
        self,
        checkpoint_path,
        transformer_model,
        mamba_config,
        attn_layers,
        dtype,
        load_from_hub=False,
        **kwargs,
    ):
        super(MambaTransformerHybridModelWrapper, self).__init__()
        self.mamba_config = mamba_config
        self.attn_layers = attn_layers
        self.ssm_layers = [
            layer_idx
            for layer_idx in range(mamba_config.n_layer)
            if layer_idx not in attn_layers
        ]
        self.model = transformer_model
        self.config = self.model.config
        for layer_idx in range(mamba_config.n_layer):
            if layer_idx in attn_layers:
                layer_encoder = MHADecoderLayer(
                    self.config,
                    layer_idx,
                    device="cuda",
                    dtype=dtype,
                )
            else:
                layer_encoder = MambaDecoderLayer(
                    mamba_config,
                    layer_idx,
                    device="cuda",
                    dtype=dtype,
                )
            self.model.model.layers[layer_idx] = layer_encoder

        if checkpoint_path is not None:
            if load_from_hub:
                # load from a huggingface hub
                ckpt = load_state_dict_hf(
                    checkpoint_path, device=torch.device("cpu"), dtype=dtype
                )
            else:
                # load from a local directory
                if os.path.exists(f"{checkpoint_path}/pytorch_model.bin"):
                    # support save from bin file
                    ckpt = torch.load(
                        f"{checkpoint_path}/pytorch_model.bin",
                        map_location=torch.device("cpu"),
                    )
                else:
                    # support save from safetensors
                    ckpt = load_safetensors_to_dict(checkpoint_path)

            if self.config.tie_word_embeddings:
                ckpt["lm_head.weight"] = ckpt["model.embed_tokens.weight"]

            self.model.load_state_dict(ckpt)

        self.model = self.model.to(dtype).cuda()
        self.device = self.model.device

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.model.model.layers)
        }

    def gradient_checkpointing_enable(self, *args, **kwargs):
        return self.model.gradient_checkpointing_enable(*args, **kwargs)

    def forward(
        self,
        input_ids,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        **mixer_kwargs,
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        if inference_params is None and num_last_tokens == 0:
            # we want to reuse hf features, so call hf forward to support gradient checkpoints
            # this is happen in the training
            attention_mask = mixer_kwargs.pop("attention_mask", None)
            return self.model(input_ids, attention_mask, position_ids, **mixer_kwargs)
        else:
            # this if for inference
            hidden_states = self.model.model.embed_tokens(input_ids)
            for decoder_layer in self.model.model.layers:
                hidden_states = decoder_layer(
                    hidden_states, inference_params=inference_params, **mixer_kwargs
                )
            hidden_states = self.model.model.norm(hidden_states)
            if num_last_tokens > 0:
                hidden_states = hidden_states[:, -num_last_tokens:]
            lm_logits = self.model.lm_head(hidden_states)
            CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
            return CausalLMOutput(logits=lm_logits)

    @staticmethod
    def from_pretrained_local(
        pretrained_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ):
        config_data = load_config_hf(pretrained_model_name)
        transformer_model = AutoModelForCausalLM.from_pretrained(
            config_data["_name_or_path"],
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        with open(f"{pretrained_model_name}/{MAMBA_CONFIG_NAME}", "r") as json_file:
            config_dict = json.load(json_file)
        mamba_config = MambaConfig(**config_dict)
        return MambaTransformerHybridModelWrapper(
            pretrained_model_name,
            transformer_model,
            mamba_config,
            mamba_config.attn_layers,
            torch_dtype,
            init_with_kqvo=False,
        )

    @staticmethod
    def from_pretrained_hub(
        pretrained_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ):
        config_data = load_config_hf(pretrained_model_name)
        transformer_model = AutoModelForCausalLM.from_pretrained(
            config_data["_name_or_path"],
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        resolved_archive_file = cached_file(
            pretrained_model_name,
            MAMBA_CONFIG_NAME,
            _raise_exceptions_for_missing_entries=False,
        )
        config_dict = json.load(open(resolved_archive_file))
        mamba_config = MambaConfig(**config_dict)
        return MambaTransformerHybridModelWrapper(
            pretrained_model_name,
            transformer_model,
            mamba_config,
            mamba_config.attn_layers,
            torch_dtype,
            init_with_kqvo=False,
            load_from_hub=True,
        )

    @staticmethod
    def from_pretrained(
        pretrained_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ):
        if os.path.exists(pretrained_model_name):
            return MambaTransformerHybridModelWrapper.from_pretrained_local(
                pretrained_model_name, torch_dtype, attn_implementation
            )
        else:
            return MambaTransformerHybridModelWrapper.from_pretrained_hub(
                pretrained_model_name, torch_dtype, attn_implementation
            )

    def save_pretrained(self, save_directory, state_dict):
        # Ensure save_directory exists
        self.save_config(save_directory)
        self.model.save_pretrained(save_directory, state_dict=state_dict)

    def save_config(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "mamba_config.json")
        with open(config_path, "w") as f:
            json.dump(self.mamba_config.__dict__, f, indent=4)

    def get_memory_footprint(self):
        return self.model.get_memory_footprint()

    def generate(
        self,
        input_ids,
        attention_mask,
        do_sample,
        max_new_tokens,
        eos_token_id,
        pad_token_id,
        generation_config,
        output_scores,
        return_dict_in_generate,
        use_cache,
        **kwargs,
    ):
        cg = kwargs.pop("cg", True)

        return super().generate(
            input_ids=input_ids,
            max_length=input_ids.shape[1] + max_new_tokens,
            cg=cg,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            temperature=generation_config.temperature,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
            eos_token_id=eos_token_id,
            **kwargs,
        )
