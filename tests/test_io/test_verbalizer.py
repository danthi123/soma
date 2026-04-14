import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch import nn

from soma.io.verbalizer import SomaAggregator, SomaVerbalizer, VerbalizerSpec


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


def test_untrained_verbalizer_emits_near_null_prefix():
    """Safety property: an untrained projector should not actively corrupt LLM
    generation. Near-zero init on the final layer means the prefix is
    essentially empty tokens, so the LLM behaves ~vanilla."""
    spec = VerbalizerSpec(
        soma_output_dim=128,
        llm_name="smoke",
        llm_hidden_dim=960,
        num_prefix_tokens=16,
    )
    v = SomaVerbalizer(spec)
    x = torch.randn(1, 128) * 10.0  # even with large input, output should be tiny
    y = v(x)
    # Final prefix should be much smaller than ~1 (the typical hidden-state
    # magnitude). Use 0.1 as a generous upper bound.
    assert y.abs().max().item() < 0.1, f"near-null init violated: max={y.abs().max().item()}"


def test_final_layer_bias_is_zero_at_init():
    spec = VerbalizerSpec(
        soma_output_dim=128,
        llm_name="smoke",
        llm_hidden_dim=512,
        num_prefix_tokens=4,
    )
    v = SomaVerbalizer(spec)
    # Find the final Linear
    linears = [m for m in v.proj if isinstance(m, nn.Linear)]
    final = linears[-1]
    assert final.bias is not None
    assert torch.all(final.bias == 0.0)


def test_save_load_round_trip(tmp_path: Path):
    spec = VerbalizerSpec(
        soma_output_dim=128,
        llm_name="smoke",
        llm_hidden_dim=960,
        num_prefix_tokens=8,
        proj_hidden_dim=256,
    )
    v = SomaVerbalizer(spec)
    # Nudge weights off init to prove round-trip
    with torch.no_grad():
        for p in v.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    out = tmp_path / "verbalizer"
    v.save(out)
    assert (out / "spec.json").exists()
    assert (out / "weights.pt").exists()

    v2 = SomaVerbalizer.load(out)
    assert v2.spec == spec
    for (n1, p1), (n2, p2) in zip(v.named_parameters(), v2.named_parameters(), strict=True):
        assert n1 == n2
        assert torch.allclose(p1, p2)


def test_load_rejects_wrong_directory(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        SomaVerbalizer.load(tmp_path / "does-not-exist")


def test_aggregator_mean_pools_output_activations():
    # 3 output nodes, each with 128-dim activation
    acts = {
        "node_a": torch.ones(128) * 1.0,
        "node_b": torch.ones(128) * 2.0,
        "node_c": torch.ones(128) * 3.0,
    }
    pooled = SomaAggregator.collapse(acts, soma_output_dim=128)
    # Mean should be 2.0 (entry-wise)
    assert pooled.shape == (1, 128)
    assert torch.allclose(pooled, torch.ones(1, 128) * 2.0)


def test_aggregator_handles_no_output_nodes():
    pooled = SomaAggregator.collapse({}, soma_output_dim=128)
    assert pooled.shape == (1, 128)
    assert torch.all(pooled == 0.0)  # zero vector when nothing is active


def test_aggregator_rejects_wrong_dim():
    acts = {"a": torch.ones(64)}
    with pytest.raises(ValueError, match="expected dim"):
        SomaAggregator.collapse(acts, soma_output_dim=128)


class _StubTextDecoder:
    """Stub mirroring the subset of TextDecoder the fallback calls."""

    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def decode_sequence(self, activations: torch.Tensor) -> str:
        self.calls.append(activations)
        return f"stub-decoded:shape={tuple(activations.shape)}"


class _StubSOMA:
    def __init__(self) -> None:
        self.text_decoder = _StubTextDecoder()


def test_fallback_text_delegates_to_soma_text_decoder():
    spec = VerbalizerSpec(
        soma_output_dim=128,
        llm_name="smoke",
        llm_hidden_dim=960,
        num_prefix_tokens=8,
    )
    v = SomaVerbalizer(spec)
    soma = _StubSOMA()
    acts = torch.randn(4, 128)
    out = v.fallback_text(soma, acts)
    assert out.startswith("stub-decoded:")
    assert len(soma.text_decoder.calls) == 1
    assert soma.text_decoder.calls[0] is acts  # no copy


def test_two_specs_independently_valid():
    # Simulate swap from SmolLM2-360M → Qwen2.5-1.5B (960 → 1536 hidden)
    spec_old = VerbalizerSpec(
        soma_output_dim=128,
        llm_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        llm_hidden_dim=960,
        num_prefix_tokens=16,
    )
    spec_new = VerbalizerSpec(
        soma_output_dim=128,
        llm_name="Qwen/Qwen2.5-1.5B-Instruct",
        llm_hidden_dim=1536,
        num_prefix_tokens=8,
    )
    v_old = SomaVerbalizer(spec_old)
    v_new = SomaVerbalizer(spec_new)

    x = torch.randn(2, 128)
    y_old = v_old(x)
    y_new = v_new(x)
    assert y_old.shape == (2, 16, 960)
    assert y_new.shape == (2, 8, 1536)
    # The two projectors are NOT interchangeable — loading one into the other
    # would fail, which is the correct property (VerbalizerSpec pins identity).


def test_cannot_load_spec_mismatch(tmp_path: Path):
    spec_a = VerbalizerSpec(
        soma_output_dim=128,
        llm_name="a",
        llm_hidden_dim=512,
        num_prefix_tokens=4,
    )
    spec_b = VerbalizerSpec(
        soma_output_dim=128,
        llm_name="b",
        llm_hidden_dim=1024,  # different!
        num_prefix_tokens=4,
    )
    v_a = SomaVerbalizer(spec_a)
    v_a.save(tmp_path / "a")

    # Hand-tamper: replace spec.json with spec_b's contents (simulating a
    # careless bundle edit)
    (tmp_path / "a" / "spec.json").write_text(json.dumps(asdict(spec_b)))
    # Now load should raise (shape mismatch) because load_state_dict
    # checks weight shapes.
    with pytest.raises((RuntimeError, ValueError)):
        SomaVerbalizer.load(tmp_path / "a")
