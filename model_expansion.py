import torch


def copy_tensor_block(src_tensor, tgt_tensor):
    """
    Copy overlapping block from src_tensor into tgt_tensor.
    Extra entries in tgt_tensor remain unchanged (or zero if pre-zeroed).
    """
    slices = tuple(slice(0, min(s, t)) for s, t in zip(src_tensor.shape, tgt_tensor.shape))
    tgt_tensor[slices] = src_tensor[slices]
    return tgt_tensor


def expand_state_dict_to_target(source_state_dict, target_model):
    """
    Expand a smaller model's state_dict into the shape of target_model.
    Shared weights are copied into the top-left overlapping block.
    Extra weights remain at target model initialization.
    """
    target_state = target_model.state_dict()

    expanded_state = {}
    for key, tgt_val in target_state.items():
        if key in source_state_dict:
            src_val = source_state_dict[key]

            if src_val.shape == tgt_val.shape:
                expanded_state[key] = src_val.clone()
            else:
                new_tensor = tgt_val.clone()
                new_tensor.zero_()
                new_tensor = copy_tensor_block(src_val, new_tensor)
                expanded_state[key] = new_tensor
        else:
            expanded_state[key] = tgt_val.clone()

    return expanded_state


def expand_model_to_large(source_model, large_model):
    """
    Expand source_model weights into large_model architecture.
    """
    source_state = source_model.state_dict()
    expanded_state = expand_state_dict_to_target(source_state, large_model)
    large_model.load_state_dict(expanded_state, strict=True)
    return large_model
