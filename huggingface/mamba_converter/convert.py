import torch
import argparse

import os
import json
import shutil
from glob import glob


from transformers import AutoTokenizer

from modeling_mamba import MambaForCausalLM, MambaConfig

COL_PARALLEL_KEYS = [
    "in_proj",
    "dt_proj",
    "embedding",
    "conv1d",
    "A_log",
    "projector",
    "lm_head",
]

KEYS_TO_IGNORE = ["norm_C", "norm_B", "norm_dt"]

CONFIG_DICT = {
    "7B": MambaConfig(
        vocab_size=65024,
        hidden_size=4096,
        state_size=16,
        num_hidden_layers=64,
        layer_norm_epsilon=1e-05,
        expand=2,
        conv_kernel=4,
        use_bias=False,
        use_conv_bias=True,
        hidden_act="silu",
        initializer_range=0.1,
        residual_in_fp32=True,
        time_step_rank="auto",
        time_step_scale=1.0,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_init_scheme="random",
        time_step_floor=1e-4,
        rescale_prenorm_residual=False,
        tie_word_embeddings=False,
    ),
    "1B_FC3": MambaConfig(
        vocab_size=131072,
        hidden_size=2048,
        state_size=16,
        num_hidden_layers=32,
        layer_norm_epsilon=1e-05,
        expand=2,
        conv_kernel=4,
        use_bias=False,
        use_conv_bias=True,
        hidden_act="silu",
        initializer_range=0.1,
        residual_in_fp32=True,
        time_step_rank="auto",
        time_step_scale=1.0,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_init_scheme="random",
        time_step_floor=1e-4,
        rescale_prenorm_residual=False,
        tie_word_embeddings=False,
    ),
    "1B": MambaConfig(
        vocab_size=130048,
        hidden_size=2048,
        state_size=16,
        num_hidden_layers=32,
        layer_norm_epsilon=1e-05,
        expand=2,
        conv_kernel=4,
        use_bias=False,
        use_conv_bias=True,
        hidden_act="silu",
        initializer_range=0.1,
        residual_in_fp32=True,
        time_step_rank="auto",
        time_step_scale=1.0,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_init_scheme="random",
        time_step_floor=1e-4,
        rescale_prenorm_residual=False,
        tie_word_embeddings=False,
    ),
    "test-64bits": MambaConfig(
        vocab_size=261120,
        hidden_size=1024,
        state_size=16,
        num_hidden_layers=24,
        layer_norm_epsilon=1e-05,
        expand=2,
        conv_kernel=4,
        use_bias=False,
        use_conv_bias=True,
        hidden_act="silu",
        initializer_range=0.1,
        residual_in_fp32=True,
        time_step_rank="auto",
        time_step_scale=1.0,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_init_scheme="random",
        time_step_floor=1e-4,
        rescale_prenorm_residual=False,
        tie_word_embeddings=False,
    ),
}


def remap_key(key):
    key = key.replace(".embedding", ".embeddings")
    key = key.replace("backbone.projector", "lm_head")

    key = key.replace(".A_log.A_log", ".A_log")

    key = key.replace(".D.D", ".D")

    return key


def is_col_parallel(key):
    return any(col_parallel_key in key for col_parallel_key in COL_PARALLEL_KEYS)


def rename_state_dict(old_state_dict):
    new_state_dict = {}
    for key, value in old_state_dict.items():
        new_key = remap_key(key)
        new_state_dict[new_key] = value
    return new_state_dict


def concat_state_dict(state_dict):
    new_state_dict = {}

    # Iterate over the state_dict and concatenate subkey tensors
    for key in state_dict:
        if ":" in key:
            # Split the key to find the base key and subkey
            base_key, subkey = key.rsplit(":", 1)

            # If the base key is not in the new_state_dict, initialize it
            if base_key not in new_state_dict:
                new_state_dict[base_key] = {}

            # Append the tensor to the list of tensors for the base key
            new_state_dict[base_key][int(subkey)] = state_dict[key]
        elif any(ignore_key in key for ignore_key in KEYS_TO_IGNORE):
            continue
        else:
            new_state_dict[key] = state_dict[key]

    # Concatenate tensors for each base key
    for base_key, tensors in new_state_dict.items():
        concat_dim = 0 if is_col_parallel(base_key) else 1

        if isinstance(tensors, dict):
            sorted_dict = dict(new_state_dict[base_key])
            new_state_dict[base_key] = torch.cat(
                list(sorted_dict.values()), dim=concat_dim
            )
    return new_state_dict


def load_and_convert_hf_model(
    state_dict, output_dir, size="7B", tokenizer_path="tokenizers/falcon2"
):
    config = CONFIG_DICT[size]

    # Init on the meta device to not allocate memory
    with torch.device("meta"):
        model = MambaForCausalLM(config)

    model.load_state_dict(state_dict, strict=True, assign=True)

    print("saving pretrained locally")
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")

    # Current file directory
    current_path = os.path.dirname(os.path.abspath(__file__))

    shutil.copy(
        os.path.join(current_path, "modeling_mamba.py"),
        os.path.join(output_dir, "modeling_mamba.py"),
    )
    json_file_path = os.path.join(output_dir, "config.json")
    with open(json_file_path, "r") as f:
        data = json.load(f)

        data["auto_map"] = {
            "AutoConfig": "modeling_mamba.MambaConfig",
            "AutoModel": "modeling_mamba.MambaModel",
            "AutoModelForCausalLM": "modeling_mamba.MambaForCausalLM",
        }

    with open(json_file_path, "w") as f:
        json.dump(data, f, indent=4)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(current_path, tokenizer_path)
    )
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert Gigatron to HF format")
    parser.add_argument("--state_dict_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--size", type=str, required=False, default="7B")
    parser.add_argument(
        "--tokenizer_path", type=str, required=False, default="tokenizers/falcon2"
    )

    args = parser.parse_args()
    state_dict_dir = args.state_dict_dir
    output_dir = args.output_dir
    size = args.size
    tokenizer_path = args.tokenizer_path

    state_dict = {}
    for state_dict_path in glob(os.path.join(state_dict_dir, "*.pt")) + glob(
        os.path.join(state_dict_dir, "**/*.pt")
    ):
        # Ignore the "_keys.pt" files
        if not state_dict_path.endswith("_keys.pt"):
            current_state_dict = torch.load(state_dict_path, map_location="cpu")
            state_dict.update(current_state_dict)

    if len(state_dict) == 0:
        raise ValueError(
            "Could not find any relevant checkpoint in the passed path. Please double check that you have passed a valid checkpoint path."
        )

    state_dict = rename_state_dict(state_dict)
    state_dict = concat_state_dict(state_dict)

    load_and_convert_hf_model(state_dict, output_dir, size, tokenizer_path)
