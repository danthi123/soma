import pytest

from soma.io.verbalizer import VerbalizerSpec


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
