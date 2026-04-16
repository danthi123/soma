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
    _cosine_lr,
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
    """Minimal HF-like tokenizer stub: supports single-text and batched
    (list-of-text) calls with optional ``padding=True``. Emits an
    ``attention_mask`` in the batched case so ``compute_lm_loss`` can mask
    out pad positions; unbatched calls omit it (matching the pre-batching
    contract used by compute_lm_loss when no mask is supplied)."""

    def __init__(self) -> None:
        self.pad_token_id = 0

    def __call__(
        self,
        text: str | list[str],
        return_tensors: str = "pt",
        padding: bool = False,
    ) -> dict:
        del return_tensors  # always pt for this stub
        if isinstance(text, str):
            ids = [min(ord(c) % 32, 31) for c in text]
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

        encoded = [[min(ord(c) % 32, 31) for c in t] for t in text]
        if padding:
            max_len = max((len(row) for row in encoded), default=0)
            padded = [row + [self.pad_token_id] * (max_len - len(row)) for row in encoded]
            attn = [[1] * len(row) + [0] * (max_len - len(row)) for row in encoded]
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            }
        return {"input_ids": torch.tensor(encoded, dtype=torch.long)}


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


# ---------------------------------------------------------------------------
# T7: eval_lm_loss — held-out measurement
# ---------------------------------------------------------------------------


def test_eval_lm_loss_returns_mean_scalar():
    t = _fresh_trainer()
    val = t.eval_lm_loss(texts=["foo", "bar", "baz qux"])
    assert isinstance(val, float)
    assert val > 0


def test_eval_lm_loss_does_not_update_verbalizer_weights():
    t = _fresh_trainer()
    before = [p.detach().clone() for p in t.verbalizer.parameters()]
    _ = t.eval_lm_loss(texts=["hello", "world"])
    after = [p.detach().clone() for p in t.verbalizer.parameters()]
    for b, a in zip(before, after, strict=True):
        assert torch.equal(b, a), "verbalizer param drifted during eval_lm_loss — no_grad missing?"


def test_eval_lm_loss_restores_train_mode():
    """After eval_lm_loss returns, the verbalizer should be back in train
    mode so subsequent train_step calls are correct."""
    t = _fresh_trainer()
    _ = t.eval_lm_loss(texts=["foo"])
    assert t.verbalizer.training, "verbalizer left in inference mode after eval_lm_loss"


def test_eval_lm_loss_after_training_is_lower():
    """Load-bearing: eval loss should drop after training on the same text.

    Uses elevated LR (same pattern as T5 loss-decrease test) to converge
    within a reasonable test budget.
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
    pre = t.eval_lm_loss(texts=[text])
    for _ in range(100):
        t.train_step(text=text)
    post = t.eval_lm_loss(texts=[text])
    assert post < pre * 0.95, f"pre={pre:.4f} post={post:.4f}"


def test_eval_lm_loss_handles_empty_texts():
    """Edge case: empty text list should return 0.0 (no division by zero)."""
    t = _fresh_trainer()
    val = t.eval_lm_loss(texts=[])
    assert val == 0.0


# ---------------------------------------------------------------------------
# Polish: train_step skips backward+optim when loss is non-finite (NaN/inf).
# Surfaced by the T9 SmolLM2 smoke test, which observed NaN training losses
# from late steps. Without this guard, a NaN gradient would corrupt Adam's
# running moments and silently poison every subsequent update.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Trainer improvements: cosine LR, grad accumulation, save-best, CSV logging
# ---------------------------------------------------------------------------


def test_cosine_lr_warmup_ramps_from_zero_to_base():
    """Linear warmup: lr(1) ≈ base/warmup, lr(warmup) ≈ base."""
    base = 1e-3
    warmup = 10
    max_steps = 100
    # Step 1 of 10-step warmup → base * 1/10.
    assert abs(_cosine_lr(1, base, max_steps, warmup) - base * 0.1) < 1e-9
    # End of warmup → full base.
    assert abs(_cosine_lr(warmup, base, max_steps, warmup) - base) < 1e-9


def test_cosine_lr_decays_to_zero_at_max_steps():
    """Half-cosine decay: lr(warmup) = base, lr(max_steps) = 0."""
    base = 1e-3
    warmup = 10
    max_steps = 100
    lr_end = _cosine_lr(max_steps, base, max_steps, warmup)
    assert lr_end < 1e-9, f"expected lr near 0 at max_steps, got {lr_end}"
    # Midpoint of decay → half of base.
    mid = warmup + (max_steps - warmup) // 2
    lr_mid = _cosine_lr(mid, base, max_steps, warmup)
    assert abs(lr_mid - base * 0.5) < 1e-3


def test_cosine_lr_no_warmup_starts_at_base():
    """warmup_steps=0 → step 1 should already be near base_lr (no ramp)."""
    base = 1e-3
    max_steps = 100
    lr_start = _cosine_lr(1, base, max_steps, 0)
    # Half-cosine at t=1/100 is very close to base.
    assert lr_start > base * 0.99


def test_train_cosine_lr_changes_over_time(tmp_path: Path):
    """With cosine_lr=True, optimizer LR should differ between early/late steps."""
    t = _fresh_trainer()
    lr_trace: list[float] = []
    original_step = t.optim.step

    def _record_step() -> None:
        lr_trace.append(t.optim.param_groups[0]["lr"])
        original_step()

    t.optim.step = _record_step  # type: ignore[method-assign]
    t.train(
        corpus=iter(["sample"] * 20),
        max_steps=10,
        out_dir=tmp_path,
        cosine_lr=True,
        warmup_steps=2,
    )
    # LR should monotonically decrease from step 2 onward (post-warmup).
    assert lr_trace[2] > lr_trace[-1], (
        f"cosine LR did not decrease: first={lr_trace[0]}, last={lr_trace[-1]}"
    )


def test_train_constant_lr_by_default(tmp_path: Path):
    """Default call (cosine_lr=False) keeps LR at config.verbalizer_lr throughout."""
    t = _fresh_trainer()
    base_lr = t.config.verbalizer_lr
    t.train(corpus=iter(["sample"] * 5), max_steps=3, out_dir=tmp_path)
    assert t.optim.param_groups[0]["lr"] == base_lr


def test_train_grad_accum_defers_optim_step(tmp_path: Path):
    """With grad_accum_steps=3, optim.step fires after every 3rd micro-batch."""
    t = _fresh_trainer()
    calls: list[int] = []
    original_step = t.optim.step

    def _counting_step() -> None:
        calls.append(len(calls) + 1)
        original_step()

    t.optim.step = _counting_step  # type: ignore[method-assign]

    t.train(
        corpus=iter(["sample"] * 10),
        max_steps=6,
        out_dir=tmp_path,
        grad_accum_steps=3,
    )
    # 6 micro-steps / 3 per optim = 2 optim.step calls.
    assert len(calls) == 2, f"expected 2 optim steps, got {len(calls)}"


def test_train_grad_accum_partial_bucket_flushed(tmp_path: Path):
    """Leftover grads from a non-full accumulation bucket must still be applied."""
    t = _fresh_trainer()
    calls: list[int] = []
    original_step = t.optim.step

    def _counting_step() -> None:
        calls.append(len(calls) + 1)
        original_step()

    t.optim.step = _counting_step  # type: ignore[method-assign]

    # 5 micro-batches with grad_accum=3: one full bucket (step 3), then
    # leftover bucket of 2 flushed at end → 2 optim.step calls total.
    t.train(
        corpus=iter(["sample"] * 10),
        max_steps=5,
        out_dir=tmp_path,
        grad_accum_steps=3,
    )
    assert len(calls) == 2, f"expected 2 optim steps (1 full + 1 flush), got {len(calls)}"


def test_train_rejects_bad_grad_accum(tmp_path: Path):
    t = _fresh_trainer()
    with pytest.raises(ValueError, match="grad_accum_steps"):
        t.train(
            corpus=iter(["sample"]),
            max_steps=1,
            out_dir=tmp_path,
            grad_accum_steps=0,
        )


def test_train_saves_best_on_eval_improvement(tmp_path: Path):
    """With eval_texts + eval_interval, a verbalizer_best/ checkpoint must
    appear after the first eval step (eval_loss starts at +inf, any real
    value improves it)."""
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
    t.train(
        corpus=iter(["the quick brown fox"] * 20),
        max_steps=10,
        out_dir=tmp_path,
        eval_texts=["some eval text"],
        eval_interval=2,
    )
    assert (tmp_path / "verbalizer_best").exists(), "best checkpoint not saved"
    assert (tmp_path / "verbalizer_best" / "weights.pt").exists()
    assert (tmp_path / "best_info.json").exists()
    import json as _json

    meta = _json.loads((tmp_path / "best_info.json").read_text(encoding="utf-8"))
    assert "step" in meta and "eval_loss" in meta


def test_train_no_best_without_eval_texts(tmp_path: Path):
    """Without eval_texts, no verbalizer_best/ should be created."""
    t = _fresh_trainer()
    t.train(
        corpus=iter(["sample"] * 5),
        max_steps=3,
        out_dir=tmp_path,
    )
    assert not (tmp_path / "verbalizer_best").exists()
    assert not (tmp_path / "best_info.json").exists()


def test_train_writes_loss_log_csv(tmp_path: Path):
    """With loss_log_path set, a CSV with per-step rows must be produced."""
    t = _fresh_trainer()
    log_path = tmp_path / "loss_log.csv"
    t.train(
        corpus=iter(["sample"] * 5),
        max_steps=4,
        out_dir=tmp_path,
        eval_texts=["eval one"],
        eval_interval=2,
        loss_log_path=log_path,
    )
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "step,train_loss,eval_loss,lr"
    assert len(lines) == 5, f"expected header + 4 rows, got {len(lines)}"
    # Row 2 (step=2) is an eval step — eval_loss field must be populated.
    row_eval = lines[2].split(",")
    assert row_eval[0] == "2"
    assert row_eval[2] != "", "eval_loss missing on eval step"
    # Row 1 (step=1) is not an eval step — eval_loss field must be empty.
    row_noeval = lines[1].split(",")
    assert row_noeval[2] == "", "eval_loss should be empty on non-eval step"


# ---------------------------------------------------------------------------
# Batching: compute_lm_loss + _step_loss over multi-sample inputs
# ---------------------------------------------------------------------------


def test_compute_lm_loss_pad_positions_do_not_affect_loss():
    """A short sample concat'd with a padded sample must yield the same loss
    as the short sample alone (padded positions masked out via -100 and
    attention_mask=0)."""
    torch.manual_seed(0)
    head = ChatHead(
        model=_TinyCausalLM(vocab=32, d_model=16),
        tokenizer=_TinyTokenizer(),
    )
    # Single-sample baseline.
    prefix_single = torch.zeros(1, 4, 16)
    tokens_single = torch.tensor([[1, 2, 3]], dtype=torch.long)
    loss_single = compute_lm_loss(
        chat_head=head,
        prefix=prefix_single,
        token_ids=tokens_single,
    ).item()

    # Two-sample batch where the second sample is entirely pad.
    prefix_batch = torch.zeros(2, 4, 16)
    tokens_batch = torch.tensor(
        [[1, 2, 3, 0, 0], [0, 0, 0, 0, 0]],
        dtype=torch.long,
    )
    attn_batch = torch.tensor(
        [[1, 1, 1, 0, 0], [0, 0, 0, 0, 0]],
        dtype=torch.long,
    )
    loss_batch = compute_lm_loss(
        chat_head=head,
        prefix=prefix_batch,
        token_ids=tokens_batch,
        token_attention_mask=attn_batch,
    ).item()

    assert abs(loss_batch - loss_single) < 1e-5, (
        f"pad positions leaked into loss: single={loss_single:.6f}, batch={loss_batch:.6f}"
    )


def test_step_loss_accepts_batch_and_returns_scalar():
    t = _fresh_trainer()
    loss_tensor, loss_value = t._step_loss(texts=["hello world", "another sample"])
    assert loss_tensor.ndim == 0
    assert isinstance(loss_value, float)
    assert loss_value > 0


def test_train_batch_updates_verbalizer_once():
    """train_batch must update the verbalizer (one optim.step per call)."""
    t = _fresh_trainer()
    before = [p.detach().clone() for p in t.verbalizer.parameters()]
    _ = t.train_batch(texts=["the quick brown fox", "jumped over"])
    after = [p.detach().clone() for p in t.verbalizer.parameters()]
    assert any(not torch.allclose(b, a) for b, a in zip(before, after, strict=True)), (
        "verbalizer did not move after train_batch"
    )


def test_train_step_delegates_to_batch_path():
    """train_step(text=...) should go through the new batched path with B=1
    and produce a sensible loss (back-compat shim for existing callers)."""
    t = _fresh_trainer()
    loss_value = t.train_step(text="hello world")
    assert isinstance(loss_value, float)
    assert loss_value > 0


def test_train_batch_size_controls_samples_per_forward(tmp_path: Path):
    """With batch_size=3 and 12 corpus items, train should do 4 forward
    passes (optim.step called 4 times — one per batch, no accumulation)."""
    t = _fresh_trainer()
    calls: list[int] = []
    original_step = t.optim.step

    def _counting_step() -> None:
        calls.append(len(calls) + 1)
        original_step()

    t.optim.step = _counting_step  # type: ignore[method-assign]

    losses = t.train(
        corpus=iter(["sample"] * 12),
        max_steps=4,
        out_dir=tmp_path,
        batch_size=3,
    )
    assert len(losses) == 4, f"expected 4 batched steps, got {len(losses)}"
    assert len(calls) == 4, f"expected 4 optim.step calls, got {len(calls)}"


def test_train_partial_tail_batch_is_still_forwarded(tmp_path: Path):
    """Corpus of 7 items with batch_size=3: batches are [3, 3, 1]. The
    partial tail (1 item) must still contribute a training step."""
    t = _fresh_trainer()
    losses = t.train(
        corpus=iter(["sample"] * 7),
        max_steps=10,
        out_dir=tmp_path,
        batch_size=3,
    )
    assert len(losses) == 3


def test_train_rejects_invalid_batch_size(tmp_path: Path):
    t = _fresh_trainer()
    with pytest.raises(ValueError, match="batch_size"):
        t.train(
            corpus=iter(["x"]),
            max_steps=1,
            out_dir=tmp_path,
            batch_size=0,
        )


def test_eval_lm_loss_batched_matches_sequential():
    """Batched eval must produce the same token-weighted mean as per-sample
    eval (within fp tolerance)."""
    t = _fresh_trainer()
    texts = ["foo", "bar baz qux", "hello world", "a"]

    # Token-weighted sequential reference.
    total_loss_tokens = 0.0
    total_tokens = 0
    for text in texts:
        one = t.eval_lm_loss(texts=[text], batch_size=1)
        # For B=1, num valid tokens = len of ids.
        ids = t.chat_head.tokenizer(text, return_tensors="pt")["input_ids"]
        n = int(ids.numel())
        total_loss_tokens += one * n
        total_tokens += n
    ref = total_loss_tokens / total_tokens

    batched = t.eval_lm_loss(texts=texts, batch_size=4)
    assert abs(batched - ref) < 1e-4, f"batched={batched:.6f} vs ref={ref:.6f}"


# ---------------------------------------------------------------------------
# Joint SOMA + verbalizer training
# ---------------------------------------------------------------------------


def test_text_to_state_trainable_returns_live_tensor(_t4_tokenizer: Any) -> None:
    """With trainable=True, the returned state must carry a grad_fn back
    into SOMA's params — the whole point of the joint-training path."""
    cfg = _soma_cfg()
    soma = SOMA(cfg, device=torch.device("cpu"))
    tokenizer, encoder = _build_text_io(cfg, _t4_tokenizer)
    state = text_to_state(
        text="hello world",
        soma=soma,
        tokenizer=tokenizer,
        encoder=encoder,
        soma_output_dim=cfg.sensor_output_dim,
        trainable=True,
    )
    assert state.requires_grad, "trainable state must require grad"
    assert state.grad_fn is not None, "trainable state must carry grad_fn"


def test_text_to_state_trainable_backprop_reaches_soma(_t4_tokenizer: Any) -> None:
    """A backward through the trainable state must populate grads on at
    least one SOMA parameter — proves the gradient chain actually lands."""
    cfg = _soma_cfg()
    soma = SOMA(cfg, device=torch.device("cpu"))
    tokenizer, encoder = _build_text_io(cfg, _t4_tokenizer)
    state = text_to_state(
        text="hello world",
        soma=soma,
        tokenizer=tokenizer,
        encoder=encoder,
        soma_output_dim=cfg.sensor_output_dim,
        trainable=True,
    )
    state.sum().backward()
    with_grad = [p for p in soma.graph.parameters() if p.grad is not None]
    assert len(with_grad) > 0, "no SOMA params received gradient — chain is broken"


def test_text_to_state_default_still_detached(_t4_tokenizer: Any) -> None:
    """Default call (trainable=False) preserves the frozen-SOMA contract."""
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


def _joint_trainer(soma_lr: float | None = None) -> VerbalizerTrainer:
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
    return VerbalizerTrainer(
        soma=soma,
        verbalizer=verbalizer,
        chat_head=chat_head,
        config=cfg,
        tokenizer=_shared_bpe_tokenizer,
        encoder=encoder,
        joint_soma=True,
        soma_lr=soma_lr,
    )


def test_joint_trainer_optim_has_two_param_groups():
    t = _joint_trainer()
    assert len(t.optim.param_groups) == 2, "joint mode needs verbalizer + SOMA groups"
    assert t.optim.param_groups[0]["lr"] == t.config.verbalizer_lr
    assert t.optim.param_groups[1]["lr"] == t.config.verbalizer_lr * 0.01


def test_joint_trainer_soma_lr_override():
    t = _joint_trainer(soma_lr=5e-5)
    assert t.optim.param_groups[1]["lr"] == 5e-5


def test_joint_train_step_updates_soma_params():
    """Load-bearing: one joint train_step must move at least one SOMA param."""
    t = _joint_trainer(soma_lr=0.01)
    soma_before = [p.detach().clone() for p in t.soma.graph.parameters()]
    _ = t.train_step(text="hello world, this is a training sample")
    soma_after = [p.detach().clone() for p in t.soma.graph.parameters()]
    moved = sum(1 for b, a in zip(soma_before, soma_after, strict=True) if not torch.equal(b, a))
    assert moved > 0, "joint training did not move any SOMA param"


def test_joint_train_step_does_not_unfreeze_chat_head():
    """Joint mode trains verbalizer + SOMA; the LLM must still stay frozen."""
    t = _joint_trainer()
    before = [p.detach().clone() for p in t.chat_head.model.parameters()]
    _ = t.train_step(text="hello world")
    after = [p.detach().clone() for p in t.chat_head.model.parameters()]
    for b, a in zip(before, after, strict=True):
        assert torch.equal(b, a), "ChatHead drifted under joint training"


def test_frozen_mode_soma_params_unchanged():
    """Control: with joint_soma=False, SOMA params stay frozen."""
    t = _fresh_trainer()
    soma_before = [p.detach().clone() for p in t.soma.graph.parameters()]
    _ = t.train_step(text="hello world")
    soma_after = [p.detach().clone() for p in t.soma.graph.parameters()]
    for b, a in zip(soma_before, soma_after, strict=True):
        assert torch.equal(b, a), (
            "SOMA param moved in default (frozen) mode — trainable path leaked"
        )


def test_train_step_skips_when_loss_is_non_finite(monkeypatch: pytest.MonkeyPatch):
    """If compute_lm_loss returns NaN, train_step must NOT call optim.step
    or backward — return the NaN as-is so callers can react."""
    t = _fresh_trainer()

    # Snapshot verbalizer params before the call.
    before = [p.detach().clone() for p in t.verbalizer.parameters()]

    # Replace the module-level compute_lm_loss with a NaN-returning stub.
    from soma.training import verbalizer_bootstrap as vb

    def _nan_loss(*, chat_head, prefix, token_ids, token_attention_mask=None):  # noqa: ARG001
        # Return a NaN scalar that has a real grad-fn (would-be-trainable)
        # so the test catches the "skip backward" guard, not "no graph".
        return (prefix.sum() * 0) + float("nan")

    monkeypatch.setattr(vb, "compute_lm_loss", _nan_loss)

    loss_value = t.train_step(text="hello")
    assert loss_value != loss_value, "expected NaN return"  # noqa: PLR0124

    # Verbalizer params must NOT have moved (no optim.step performed).
    after = [p.detach().clone() for p in t.verbalizer.parameters()]
    for b, a in zip(before, after, strict=True):
        assert torch.equal(b, a), "verbalizer drifted on NaN loss — guard not effective"
