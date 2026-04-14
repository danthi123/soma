from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from soma.core.config import SOMAConfig
from soma.io.chat_head import ChatHead
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
from soma.system import SOMA
from soma.training.verbalizer_bootstrap import (
    VerbalizerTrainer,
    compute_lm_loss,
    text_to_state,
)


# Reuse the Phase 3 mock shape — thinnest possible HF-like surface.
class _TinyCausalLM(nn.Module):
    """Mock HF causal LM with minimal causal mixing so the prefix actually
    conditions later-position predictions.

    Without any cross-position mixing, the verbalizer prefix could only
    affect the prediction at position k-1 (first real token). Learning
    would plateau at the uniform-distribution log-loss minus 1/T, which
    makes the end-to-end gradient test artificially capped. A single
    causal-mean-pool layer before lm_head gives every token position
    access to the prefix signal, matching the qualitative behavior of
    a real transformer's causal attention.
    """

    def __init__(self, vocab: int = 32, d_model: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.lm_head = nn.Linear(d_model, vocab)
        self.config = type("Cfg", (), {"hidden_size": d_model, "vocab_size": vocab})()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def _causal_mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d). Return y where y[:, t] = mean(x[:, :t+1]).
        # This is the simplest causal aggregator — zero parameters, but
        # each position sees all prior (including prefix) positions, so
        # the verbalizer prefix conditions every prediction downstream.
        cumsum = torch.cumsum(x, dim=1)
        lengths = torch.arange(1, x.shape[1] + 1, device=x.device).view(1, -1, 1)
        return cumsum / lengths

    def forward(self, *, inputs_embeds, attention_mask=None, labels=None, **_):
        mixed = self._causal_mean_pool(inputs_embeds)
        logits = self.lm_head(mixed)
        if labels is None:
            return type("Out", (), {"logits": logits, "loss": None})()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return type("Out", (), {"logits": logits, "loss": loss})()


class _TinyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        ids = [min(ord(c) % 32, 31) for c in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def _soma_cfg() -> SOMAConfig:
    return SOMAConfig(
        sensor_output_dim=8,
        associator_input_dim=8,
        associator_hidden_dim=16,
        associator_output_dim=8,
        integrator_input_dim=16,
        integrator_hidden_dim=16,
        integrator_output_dim=16,
        position_dim=4,
        wm_slots=2,
        wm_dim=8,
        episodic_capacity=4,
        key_dim=8,
        value_dim=8,
        vocab_size=16,
        text_embed_dim=8,
        max_nodes=32,
        initial_associator_count=2,
        initial_integrator_count=1,
        max_input_tokens=8,
        max_output_tokens=4,
        seed=0,
    )


# Module-scoped BPE tokenizer cached for reuse across _fresh_trainer calls.
# Training a tokenizer is the slowest per-test cost; building it once at
# module import keeps the full suite fast.
_BOOTSTRAP_CORPUS: tuple[str, ...] = (
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "foo bar baz qux quux",
    "soma learns to speak by listening",
    "tokens become embeddings which become activations",
) * 4
_shared_bpe_tokenizer: Any = train_bpe_tokenizer(list(_BOOTSTRAP_CORPUS), vocab_size=128)


def _fresh_trainer() -> VerbalizerTrainer:
    cfg = _soma_cfg()
    soma = SOMA(cfg, device=torch.device("cpu"))
    # OUTPUT nodes in the seed graph are sized to sensor_output_dim (see
    # SOMA._initialize_seed_graph), so that's the dim the collapsed state
    # carries into the verbalizer — NOT integrator_output_dim.
    spec = VerbalizerSpec(
        soma_output_dim=cfg.sensor_output_dim,
        llm_name="mock",
        llm_hidden_dim=16,
        num_prefix_tokens=4,
        proj_hidden_dim=16,
    )
    verbalizer = SomaVerbalizer(spec)
    chat_head = ChatHead(
        model=_TinyCausalLM(vocab=32, d_model=16),
        tokenizer=_TinyTokenizer(),
    )
    encoder = TextEncoder(
        _shared_bpe_tokenizer,
        embed_dim=cfg.text_embed_dim,
        max_seq_len=cfg.max_input_tokens,
    )
    return VerbalizerTrainer(
        soma=soma,
        verbalizer=verbalizer,
        chat_head=chat_head,
        config=cfg,
        tokenizer=_shared_bpe_tokenizer,
        encoder=encoder,
    )


def test_trainer_stores_components():
    t = _fresh_trainer()
    assert t.soma is not None
    assert t.verbalizer is not None
    assert t.chat_head is not None


def test_trainer_creates_adam_on_verbalizer_only():
    t = _fresh_trainer()
    assert isinstance(t.optim, torch.optim.Adam)
    verb_param_ids = {id(p) for p in t.verbalizer.parameters()}
    for group in t.optim.param_groups:
        for p in group["params"]:
            assert id(p) in verb_param_ids, "optimizer contains non-verbalizer param"


def test_trainer_does_not_unfreeze_chat_head():
    t = _fresh_trainer()
    assert not any(p.requires_grad for p in t.chat_head.model.parameters())


def test_trainer_uses_config_lr():
    t = _fresh_trainer()
    assert t.optim.param_groups[0]["lr"] == t.config.verbalizer_lr


def test_trainer_raises_if_chat_head_is_not_frozen():
    """Defensive: if someone hands in a ChatHead whose params got unfrozen,
    the trainer must refuse to start rather than silently train the LLM."""
    cfg = _soma_cfg()
    soma = SOMA(cfg, device=torch.device("cpu"))
    spec = VerbalizerSpec(
        soma_output_dim=cfg.sensor_output_dim,
        llm_name="mock",
        llm_hidden_dim=16,
        num_prefix_tokens=4,
        proj_hidden_dim=16,
    )
    verbalizer = SomaVerbalizer(spec)
    chat_head = ChatHead(
        model=_TinyCausalLM(vocab=32, d_model=16),
        tokenizer=_TinyTokenizer(),
    )
    encoder = TextEncoder(
        _shared_bpe_tokenizer,
        embed_dim=cfg.text_embed_dim,
        max_seq_len=cfg.max_input_tokens,
    )
    # Poison: unfreeze the model after ChatHead init.
    for p in chat_head.model.parameters():
        p.requires_grad = True

    with pytest.raises(ValueError, match="frozen"):
        VerbalizerTrainer(
            soma=soma,
            verbalizer=verbalizer,
            chat_head=chat_head,
            config=cfg,
            tokenizer=_shared_bpe_tokenizer,
            encoder=encoder,
        )


def test_compute_lm_loss_returns_scalar_tensor():
    prefix = torch.randn(1, 4, 16, requires_grad=True)
    token_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    head = ChatHead(
        model=_TinyCausalLM(vocab=32, d_model=16),
        tokenizer=_TinyTokenizer(),
    )
    loss = compute_lm_loss(chat_head=head, prefix=prefix, token_ids=token_ids)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_compute_lm_loss_gradient_reaches_prefix_only():
    prefix = torch.randn(1, 4, 16, requires_grad=True)
    token_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    head = ChatHead(
        model=_TinyCausalLM(vocab=32, d_model=16),
        tokenizer=_TinyTokenizer(),
    )
    loss = compute_lm_loss(chat_head=head, prefix=prefix, token_ids=token_ids)
    loss.backward()
    assert prefix.grad is not None
    assert torch.any(prefix.grad != 0), "prefix got zero gradient — loss path broken"
    # Frozen LLM params must stay grad-free.
    assert all(p.grad is None for p in head.model.parameters())


def test_compute_lm_loss_masks_prefix_positions():
    """Loss must ignore prefix-position predictions (labels=-100 contract).

    With a zero-valued prefix, k=4 vs k=0 should produce near-identical
    token-position loss (prefix zero vectors don't change logits). If
    prefix positions were NOT masked, loss_k4 would include k=4 extra
    uniform-random-prediction positions of significant extra loss.
    """
    torch.manual_seed(0)
    token_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    head = ChatHead(
        model=_TinyCausalLM(vocab=32, d_model=16),
        tokenizer=_TinyTokenizer(),
    )

    prefix_k4 = torch.zeros(1, 4, 16)  # zero prefix, no meaningful content
    prefix_k0 = torch.zeros(1, 0, 16)  # empty prefix

    loss_k4 = compute_lm_loss(chat_head=head, prefix=prefix_k4, token_ids=token_ids)
    loss_k0 = compute_lm_loss(chat_head=head, prefix=prefix_k0, token_ids=token_ids)

    # Same per-token loss — delta must be small (< ~0.3 accounting for
    # marginal effects of the zero prefix passing through attention).
    assert abs(loss_k4.item() - loss_k0.item()) < 0.3, (
        f"k=4 vs k=0 loss differ by {abs(loss_k4.item() - loss_k0.item()):.3f} — "
        "prefix positions may not be masked with -100"
    )


# ---------------------------------------------------------------------------
# T4: text_to_state — tokenize + embed + soma.step (no-grad) + collapse
# ---------------------------------------------------------------------------


# Shared tokenizer for the T4 tests. Trained once over a tiny corpus; the
# tokenizer is deterministic so re-use across tests is safe and fast.
_T4_CORPUS: tuple[str, ...] = (
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "foo bar baz qux quux",
    "soma learns to speak by listening",
    "tokens become embeddings which become activations",
) * 4


@pytest.fixture(scope="module")
def _t4_tokenizer() -> Any:
    return train_bpe_tokenizer(list(_T4_CORPUS), vocab_size=128)


def _build_text_io(cfg: SOMAConfig, tokenizer: Any) -> tuple[Any, TextEncoder]:
    """Build a (tokenizer, encoder) pair sized for the tiny test SOMA.

    The encoder's ``embed_dim`` must equal ``cfg.sensor_output_dim`` so the
    per-token embedding flows directly into the text sensor node (the text
    sensor has ``input_dim=cfg.sensor_output_dim``). ``cfg.text_embed_dim``
    is set equal to ``sensor_output_dim`` in ``_soma_cfg`` above.
    """
    encoder = TextEncoder(
        tokenizer,
        embed_dim=cfg.text_embed_dim,
        max_seq_len=cfg.max_input_tokens,
    )
    return tokenizer, encoder


def test_text_to_state_returns_pooled_tensor(_t4_tokenizer: Any) -> None:
    cfg = _soma_cfg()
    soma = SOMA(cfg, device=torch.device("cpu"))
    tokenizer, encoder = _build_text_io(cfg, _t4_tokenizer)
    # Use sensor_output_dim: OUTPUT nodes in the tiny test SOMA are sized to
    # sensor_output_dim (see _initialize_seed_graph in system.py), so that's
    # the dim SomaAggregator.collapse will see from last_activation.
    state = text_to_state(
        text="hello world",
        soma=soma,
        tokenizer=tokenizer,
        encoder=encoder,
        soma_output_dim=cfg.sensor_output_dim,
    )
    assert state.shape == (1, cfg.sensor_output_dim)


def test_text_to_state_deterministic_on_same_input(_t4_tokenizer: Any) -> None:
    """Two fresh SOMAs (same seed) + same text yield the same pooled state.

    We build two independent SOMAs (rather than calling text_to_state twice
    on the same SOMA) because soma.step mutates internal state (global_step,
    working memory, homeostasis, last_activation on every node) every call.
    Two consecutive invocations on one SOMA would see different working-memory
    / global_step contexts and therefore different outputs — that's a property
    of SOMA's statefulness, not a bug in text_to_state.
    """
    tokenizer = _t4_tokenizer
    cfg_a = _soma_cfg()
    soma_a = SOMA(cfg_a, device=torch.device("cpu"))
    _, encoder_a = _build_text_io(cfg_a, tokenizer)
    cfg_b = _soma_cfg()
    soma_b = SOMA(cfg_b, device=torch.device("cpu"))
    _, encoder_b = _build_text_io(cfg_b, tokenizer)

    s1 = text_to_state(
        text="foo",
        soma=soma_a,
        tokenizer=tokenizer,
        encoder=encoder_a,
        soma_output_dim=cfg_a.sensor_output_dim,
    )
    s2 = text_to_state(
        text="foo",
        soma=soma_b,
        tokenizer=tokenizer,
        encoder=encoder_b,
        soma_output_dim=cfg_b.sensor_output_dim,
    )
    assert torch.allclose(s1, s2)


def test_text_to_state_does_not_train_soma(_t4_tokenizer: Any) -> None:
    """The load-bearing invariant: no SOMA learnable parameter drifts.

    Snapshots every node parameter before the call, runs the helper on a
    non-trivial input, then re-checks each snapshot. Any drift proves either
    the no_grad wrapper is missing or eval_mode is leaking into the growth /
    update_step path.
    """
    cfg = _soma_cfg()
    soma = SOMA(cfg, device=torch.device("cpu"))
    tokenizer, encoder = _build_text_io(cfg, _t4_tokenizer)

    # Snapshot the full set of node parameters, keyed by (node_id, param_name).
    before: dict[tuple[str, str], torch.Tensor] = {}
    for node_id, node in soma.graph.nodes.items():
        for pname, p in node.named_parameters():
            before[(node_id, pname)] = p.detach().clone()

    _ = text_to_state(
        text="hello world, this is a somewhat longer input",
        soma=soma,
        tokenizer=tokenizer,
        encoder=encoder,
        soma_output_dim=cfg.sensor_output_dim,
    )

    for node_id, node in soma.graph.nodes.items():
        for pname, p in node.named_parameters():
            key = (node_id, pname)
            if key not in before:
                # Node was added mid-call (shouldn't happen with eval_mode,
                # but guard anyway — any new node is a failure).
                pytest.fail(f"Unexpected new node parameter {key} — growth fired under no_grad?")
            assert torch.allclose(before[key], p.detach()), (
                f"SOMA parameter {key} drifted during text_to_state — "
                "no_grad wrapper or eval_mode missing?"
            )


def test_text_to_state_empty_string_returns_zero_vector(_t4_tokenizer: Any) -> None:
    """Empty text → zero SOMA steps → SomaAggregator.collapse returns zeros.

    TextEncoder.encode("") returns [], so no token is ever fed to SOMA.
    OUTPUT-node last_activation stays None, _current_output_activations
    returns {}, and SomaAggregator.collapse falls back to a zero vector.
    """
    cfg = _soma_cfg()
    soma = SOMA(cfg, device=torch.device("cpu"))
    tokenizer, encoder = _build_text_io(cfg, _t4_tokenizer)
    state = text_to_state(
        text="",
        soma=soma,
        tokenizer=tokenizer,
        encoder=encoder,
        soma_output_dim=cfg.sensor_output_dim,
    )
    assert state.shape == (1, cfg.sensor_output_dim)
    assert torch.all(state == 0)


def test_text_to_state_returns_detached_tensor(_t4_tokenizer: Any) -> None:
    """Returned tensor must not carry a grad_fn — verbalizer path grafts
    its own autograd subgraph on top; a stray SOMA grad_fn would chain
    gradients back into SOMA's (frozen) parameters on ``.backward()``."""
    cfg = _soma_cfg()
    soma = SOMA(cfg, device=torch.device("cpu"))
    tokenizer, encoder = _build_text_io(cfg, _t4_tokenizer)
    state = text_to_state(
        text="hello world",
        soma=soma,
        tokenizer=tokenizer,
        encoder=encoder,
        soma_output_dim=cfg.sensor_output_dim,
    )
    assert not state.requires_grad
    assert state.grad_fn is None


# ---------------------------------------------------------------------------
# T5: train_step — full forward + backward + optim.step through the verbalizer
# ---------------------------------------------------------------------------


def test_train_step_returns_float_loss():
    t = _fresh_trainer()
    loss = t.train_step(text="hello")
    assert isinstance(loss, float)
    assert loss > 0


def test_train_step_updates_verbalizer_weights():
    t = _fresh_trainer()
    before = [p.detach().clone() for p in t.verbalizer.parameters()]
    _ = t.train_step(text="hello world, this is a test sample")
    after = [p.detach().clone() for p in t.verbalizer.parameters()]
    assert any(not torch.allclose(b, a) for b, a in zip(before, after, strict=True)), (
        "no verbalizer param moved after train_step"
    )


def test_train_step_leaves_chat_head_bit_exact():
    t = _fresh_trainer()
    before = [p.detach().clone() for p in t.chat_head.model.parameters()]
    _ = t.train_step(text="hello world")
    after = [p.detach().clone() for p in t.chat_head.model.parameters()]
    for b, a in zip(before, after, strict=True):
        assert torch.equal(b, a), "ChatHead param drifted — LLM not frozen"


def test_train_step_loss_decreases_over_iterations():
    """Load-bearing: 50 steps on the same sample should reduce loss.

    Proves the whole gradient chain (text → SOMA state → verbalizer prefix
    → LLM loss → backward → optim.step) actually wires together end-to-end.

    Uses an elevated ``verbalizer_lr`` (0.01) to converge in 50 steps for
    test speed. The production default (1e-4) is tuned for a 360M-param
    LLM where tiny per-step updates compound over thousands of samples;
    on the 32-vocab x 16-d mock here the signal is too small per step for
    that LR to move the needle in 50 iterations. The end-to-end gradient
    chain is what's load-bearing, not the specific LR.
    """
    cfg = replace(_soma_cfg(), verbalizer_lr=0.01)
    soma = SOMA(cfg, device=torch.device("cpu"))
    spec = VerbalizerSpec(
        soma_output_dim=cfg.sensor_output_dim,
        llm_name="mock",
        llm_hidden_dim=16,
        num_prefix_tokens=4,
        proj_hidden_dim=16,
    )
    verbalizer = SomaVerbalizer(spec)
    chat_head = ChatHead(
        model=_TinyCausalLM(vocab=32, d_model=16),
        tokenizer=_TinyTokenizer(),
    )
    encoder = TextEncoder(
        _shared_bpe_tokenizer,
        embed_dim=cfg.text_embed_dim,
        max_seq_len=cfg.max_input_tokens,
    )
    t = VerbalizerTrainer(
        soma=soma,
        verbalizer=verbalizer,
        chat_head=chat_head,
        config=cfg,
        tokenizer=_shared_bpe_tokenizer,
        encoder=encoder,
    )
    text = "the quick brown fox jumps over the lazy dog"
    first_loss = t.train_step(text=text)
    # 99 more iterations — 50 left only ~0.13% margin vs the 5% threshold
    # (deterministic 0.9487 ratio). 100 steps converges further without
    # changing the semantic guarantee ("loss actually drops").
    for _ in range(99):
        t.train_step(text=text)
    final_loss = t.train_step(text=text)
    assert final_loss < first_loss * 0.95, (
        f"final_loss={final_loss:.4f} not sufficiently below first_loss={first_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# T6: train — outer loop with intermediate + final checkpoints
# ---------------------------------------------------------------------------


def test_train_runs_to_max_steps(tmp_path: Path):
    t = _fresh_trainer()
    corpus = ["hello world"] * 10
    losses = t.train(
        corpus=iter(corpus),
        max_steps=5,
        out_dir=tmp_path,
    )
    assert len(losses) == 5
    assert all(isinstance(loss, float) for loss in losses)


def test_train_stops_early_on_corpus_exhaustion(tmp_path: Path):
    t = _fresh_trainer()
    # Corpus has 3 samples; max_steps=10 — loop should exit after 3.
    corpus = ["sample a", "sample b", "sample c"]
    losses = t.train(
        corpus=iter(corpus),
        max_steps=10,
        out_dir=tmp_path,
    )
    assert len(losses) == 3


def test_train_saves_final_verbalizer_checkpoint(tmp_path: Path):
    t = _fresh_trainer()
    t.train(
        corpus=iter(["hello world"] * 20),
        max_steps=3,
        out_dir=tmp_path,
    )
    assert (tmp_path / "verbalizer_final").exists()
    assert (tmp_path / "verbalizer_final" / "spec.json").exists()
    assert (tmp_path / "verbalizer_final" / "weights.pt").exists()


def test_train_saves_intermediate_checkpoints(tmp_path: Path):
    """With checkpoint_interval=2 and max_steps=5: save at step 2, step 4,
    and final."""
    # Build a trainer with checkpoint_interval=2.
    cfg = replace(_soma_cfg(), verbalizer_checkpoint_interval=2)
    soma = SOMA(cfg, device=torch.device("cpu"))
    spec = VerbalizerSpec(
        soma_output_dim=cfg.sensor_output_dim,
        llm_name="mock",
        llm_hidden_dim=16,
        num_prefix_tokens=4,
        proj_hidden_dim=16,
    )
    verbalizer = SomaVerbalizer(spec)
    chat_head = ChatHead(
        model=_TinyCausalLM(vocab=32, d_model=16),
        tokenizer=_TinyTokenizer(),
    )
    encoder = TextEncoder(
        _shared_bpe_tokenizer,
        embed_dim=cfg.text_embed_dim,
        max_seq_len=cfg.max_input_tokens,
    )
    t = VerbalizerTrainer(
        soma=soma,
        verbalizer=verbalizer,
        chat_head=chat_head,
        config=cfg,
        tokenizer=_shared_bpe_tokenizer,
        encoder=encoder,
    )
    t.train(
        corpus=iter(["sample"] * 10),
        max_steps=5,
        out_dir=tmp_path,
    )
    assert (tmp_path / "verbalizer_step_2").exists(), "missing step-2 checkpoint"
    assert (tmp_path / "verbalizer_step_4").exists(), "missing step-4 checkpoint"
    assert (tmp_path / "verbalizer_final").exists(), "missing final checkpoint"
    # Step 5 is NOT a checkpoint boundary (5 % 2 != 0).
    assert not (tmp_path / "verbalizer_step_5").exists()
