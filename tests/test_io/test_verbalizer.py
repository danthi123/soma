import pytest
import torch

from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec


def test_spec_defaults_and_frozen():
    spec = VerbalizerSpec(
        soma_output_dim=128,
        llm_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        llm_hidden_dim=960,
        num_prefix_tokens=16,
    )
    assert spec.soma_output_dim == 128
    assert spec.llm_name == "HuggingFaceTB/SmolLM2-360M-Instruct"
    assert spec.llm_hidden_dim == 960
    assert spec.num_prefix_tokens == 16
    assert spec.proj_hidden_dim == 512  # default

    # Frozen — mutation should raise
    with pytest.raises((AttributeError, TypeError)):
        spec.num_prefix_tokens = 32  # type: ignore[misc]


def test_spec_validates_positive_dims():
    with pytest.raises(ValueError, match="positive"):
        VerbalizerSpec(
            soma_output_dim=0,
            llm_name="x",
            llm_hidden_dim=960,
            num_prefix_tokens=16,
        )
    with pytest.raises(ValueError, match="positive"):
        VerbalizerSpec(
            soma_output_dim=128,
            llm_name="x",
            llm_hidden_dim=0,
            num_prefix_tokens=16,
        )
    with pytest.raises(ValueError, match="positive"):
        VerbalizerSpec(
            soma_output_dim=128,
            llm_name="x",
            llm_hidden_dim=960,
            num_prefix_tokens=0,
        )


def test_verbalizer_constructs_with_expected_param_count():
    spec = VerbalizerSpec(
        soma_output_dim=128,
        llm_name="smoke",
        llm_hidden_dim=960,
        num_prefix_tokens=16,
        proj_hidden_dim=512,
    )
    v = SomaVerbalizer(spec)
    # Check the two linear layers exist and have expected shapes
    params = dict(v.named_parameters())
    assert len(params) > 0
    # Exact names don't matter; count trainable params.
    total = sum(p.numel() for p in v.parameters() if p.requires_grad)
    # Rough upper bound: (128*512 + 512) + (512*16*960 + 16*960) ≈ 7.9M
    assert total < 10_000_000
    assert total > 5_000_000


def test_verbalizer_stores_spec():
    spec = VerbalizerSpec(
        soma_output_dim=128,
        llm_name="smoke",
        llm_hidden_dim=960,
        num_prefix_tokens=8,
    )
    v = SomaVerbalizer(spec)
    assert v.spec == spec
    assert v.spec is spec  # frozen — same instance


def test_forward_2d_input_produces_prefix():
    spec = VerbalizerSpec(
        soma_output_dim=128,
        llm_name="smoke",
        llm_hidden_dim=960,
        num_prefix_tokens=8,
    )
    v = SomaVerbalizer(spec)
    x = torch.randn(4, 128)
    y = v(x)
    assert y.shape == (4, 8, 960)


def test_forward_1d_input_auto_unsqueezes():
    spec = VerbalizerSpec(
        soma_output_dim=128,
        llm_name="smoke",
        llm_hidden_dim=512,
        num_prefix_tokens=4,
    )
    v = SomaVerbalizer(spec)
    x = torch.randn(128)  # no batch dim
    y = v(x)
    assert y.shape == (1, 4, 512)


def test_forward_rejects_wrong_last_dim():
    spec = VerbalizerSpec(
        soma_output_dim=128,
        llm_name="smoke",
        llm_hidden_dim=512,
        num_prefix_tokens=4,
    )
    v = SomaVerbalizer(spec)
    x = torch.randn(2, 64)  # wrong last dim
    with pytest.raises(ValueError, match="soma_output_dim"):
        v(x)
