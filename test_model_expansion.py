import torch
from cnn_heterogeneous_model import build_small_model, build_medium_model, build_large_model
from model_expansion import expand_model_to_large


def tensors_match_in_overlap(src, tgt):
    slices = tuple(slice(0, min(s, t)) for s, t in zip(src.shape, tgt.shape))
    return torch.allclose(src[slices], tgt[slices])


def test_small_to_large_expansion():
    small = build_small_model()
    large = build_large_model()

    # Fill small model weights with known values
    for name, param in small.named_parameters():
        torch.nn.init.constant_(param, 1.2345)

    expanded_large = expand_model_to_large(small, large)

    small_state = small.state_dict()
    large_state = expanded_large.state_dict()

    for key in small_state:
        assert key in large_state, f"Missing key in large model: {key}"
        assert tensors_match_in_overlap(small_state[key], large_state[key]), f"Mismatch in {key}"

    print("Small-to-large expansion test passed.")


def test_medium_to_large_expansion():
    medium = build_medium_model()
    large = build_large_model()

    for name, param in medium.named_parameters():
        torch.nn.init.constant_(param, 2.3456)

    expanded_large = expand_model_to_large(medium, large)

    medium_state = medium.state_dict()
    large_state = expanded_large.state_dict()

    for key in medium_state:
        assert key in large_state, f"Missing key in large model: {key}"
        assert tensors_match_in_overlap(medium_state[key], large_state[key]), f"Mismatch in {key}"

    print("Medium-to-large expansion test passed.")


if __name__ == "__main__":
    test_small_to_large_expansion()
    test_medium_to_large_expansion()
    print("All expansion unit tests passed.")
