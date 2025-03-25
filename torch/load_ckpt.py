from safetensors import safe_open


def list_parameter_keys(safetensors_path):
    """List all parameter keys available in the SafeTensors file."""
    print(f"Listing parameter keys from: {safetensors_path}")

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        print("\nAvailable parameter keys:")
        for i, key in enumerate(keys, 1):
            print(f"{i}. {key}")

    return keys


def load_selected_parameters(safetensors_path, selected_keys):
    """Load only the selected parameters from the SafeTensors file."""
    selected_params = {}

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in selected_keys:
            if key in f.keys():
                selected_params[key] = f.get_tensor(key)
                print(f"Loaded: {key} (shape: {selected_params[key].shape})")
            else:
                print(f"Warning: Key '{key}' not found in the model file")

    return selected_params


def main():
    model_path = ""
    selected_keys = [""]
    print(f"\nSelected parameters: {selected_keys}")

    # Load only the selected parameters
    loaded_params = load_selected_parameters(model_path, selected_keys)

    print("\nDone! The selected parameters have been loaded.")
    print(f"Total parameters loaded: {len(loaded_params)}")

    # Example of how to access the loaded parameters
    # for key, tensor in loaded_params.items():
    #     print(f"{key}: {tensor.shape}")


if __name__ == "__main__":
    main()
