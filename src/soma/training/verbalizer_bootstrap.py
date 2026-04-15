"""Bootstrap training loop for the SomaVerbalizer.

Only the verbalizer's projector trains. SOMA stays frozen (via
``torch.no_grad`` during state production — enforced in later tasks),
and the ChatHead's LLM stays frozen (via Phase 3's ``requires_grad=False``
contract — enforced at trainer init).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import torch

from soma.core.config import SOMAConfig

_log = logging.getLogger(__name__)


class VerbalizerTrainer:
    """Bootstrap-trains a SomaVerbalizer against a frozen LLM via LM loss.

    The only trainable parameters are the verbalizer's; SOMA's Hebbian
    graph and the ChatHead's LLM both stay fixed by construction.
    """

    def __init__(
        self,
        *,
        soma: Any,
        verbalizer: Any,
        chat_head: Any,
        config: SOMAConfig,
        tokenizer: Any,
        encoder: Any,
    ) -> None:
        self.soma = soma
        self.verbalizer = verbalizer
        self.chat_head = chat_head
        self.config = config
        # Tokenizer + encoder are plumbed here (rather than pulled off the
        # SOMA instance) because SOMA itself has no default .tokenizer /
        # .text_encoder attributes. Storing them underscore-prefixed to
        # signal "internal plumbing, not user-facing trainer surface".
        self._tokenizer = tokenizer
        self._encoder = encoder

        # Sanity: ChatHead must already be frozen (Phase 3 invariant).
        if any(p.requires_grad for p in self.chat_head.model.parameters()):
            raise ValueError(
                "ChatHead model parameters must be frozen before bootstrap "
                "training. Construct ChatHead via its __init__ (which sets "
                "requires_grad=False on every param) and do not unfreeze."
            )

        # Adam on verbalizer only — SOMA has its own Hebbian learning path
        # that runs inside soma.step(); that path must not be reached during
        # bootstrap (text_to_state in T4 wraps soma.step in torch.no_grad).
        self.optim = torch.optim.Adam(
            self.verbalizer.parameters(),
            lr=config.verbalizer_lr,
        )

    def train_step(self, *, text: str) -> float:
        """One forward + backward + optim.step. Returns scalar loss as float.

        Gradient flows only through the verbalizer's two Linear layers.
        SOMA is wrapped in ``torch.no_grad`` inside ``text_to_state`` (T4);
        ChatHead is frozen via the invariant enforced at ``__init__`` (T2).
        """
        self.verbalizer.train()
        self.optim.zero_grad()

        state = text_to_state(
            text=text,
            soma=self.soma,
            tokenizer=self._tokenizer,
            encoder=self._encoder,
            soma_output_dim=self.verbalizer.spec.soma_output_dim,
        )
        prefix = self.verbalizer(state)

        tok_out = self.chat_head.tokenizer(text, return_tensors="pt")
        token_ids = tok_out["input_ids"]

        loss = compute_lm_loss(
            chat_head=self.chat_head,
            prefix=prefix,
            token_ids=token_ids,
        )

        loss_value = float(loss.item())
        # Skip backward + optim.step on non-finite loss so a single
        # degenerate sample (NaN/inf SOMA state, exploding logits, etc.)
        # cannot corrupt Adam's running moments and silently poison every
        # subsequent update. Caller still sees the NaN in returned losses
        # and can react (skip-ahead, lower LR, dump checkpoint, etc.).
        if not math.isfinite(loss_value):
            _log.warning(
                "verbalizer train_step skipped: non-finite loss %r on "
                "text=%r — no backward/optim.step performed.",
                loss_value,
                text[:60],
            )
            return loss_value

        loss.backward()  # type: ignore[no-untyped-call]
        self.optim.step()
        return loss_value

    def train(
        self,
        *,
        corpus: Iterable[str],
        max_steps: int,
        out_dir: Path,
    ) -> list[float]:
        """Run up to ``max_steps`` train_step calls over ``corpus``.

        Saves an intermediate verbalizer checkpoint every
        ``self.config.verbalizer_checkpoint_interval`` steps (directory named
        ``verbalizer_step_N``), plus a final checkpoint named
        ``verbalizer_final``. Returns the per-step loss list.

        Exits early if the corpus iterator is exhausted before ``max_steps``;
        the final checkpoint is still written to mark where training stopped.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        losses: list[float] = []
        corpus_iter = iter(corpus)
        interval = self.config.verbalizer_checkpoint_interval

        for step in range(1, max_steps + 1):
            try:
                text = next(corpus_iter)
            except StopIteration:
                break
            loss = self.train_step(text=text)
            losses.append(loss)
            if step % interval == 0:
                self.verbalizer.save(out_dir / f"verbalizer_step_{step}")

        self.verbalizer.save(out_dir / "verbalizer_final")
        return losses

    @torch.no_grad()
    def eval_lm_loss(self, *, texts: list[str]) -> float:
        """Mean LM loss over held-out texts. No gradient, no optim step.

        Puts the verbalizer in inference mode (``.train(False)``) for the
        duration, then restores training mode before returning — so callers
        can freely alternate eval and train calls without manual bookkeeping.
        """
        self.verbalizer.train(False)
        try:
            total = 0.0
            count = 0
            for text in texts:
                state = text_to_state(
                    text=text,
                    soma=self.soma,
                    tokenizer=self._tokenizer,
                    encoder=self._encoder,
                    soma_output_dim=self.verbalizer.spec.soma_output_dim,
                )
                prefix = self.verbalizer(state)
                tok_out = self.chat_head.tokenizer(text, return_tensors="pt")
                loss = compute_lm_loss(
                    chat_head=self.chat_head,
                    prefix=prefix,
                    token_ids=tok_out["input_ids"],
                )
                total += float(loss.item())
                count += 1
        finally:
            self.verbalizer.train(True)
        return total / count if count > 0 else 0.0


def compute_lm_loss(
    *,
    chat_head: Any,
    prefix: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Causal-LM cross-entropy conditioned on a continuous prefix.

    ``prefix`` is the (B, k, d_model) soft-prompt from the verbalizer;
    ``token_ids`` is the (B, T) target text. Returns a scalar loss.

    Prefix positions are masked via ``-100`` labels so only token-position
    predictions contribute to the loss. Gradient flows through ``prefix``
    back into the verbalizer; the LLM stays frozen (no grads recorded).
    """
    batch_size, num_prefix, _ = prefix.shape
    _, num_tokens = token_ids.shape

    # Align token_ids to the LLM's device before embedding (HF tokenizers
    # return CPU tensors regardless of where the model lives). Mirrors
    # the Phase 7 T6 fix in ChatSession.respond / SOMA.chat.
    llm_device = chat_head.model.get_input_embeddings().weight.device
    if token_ids.device != llm_device:
        token_ids = token_ids.to(llm_device)

    token_embeds = chat_head.model.get_input_embeddings()(token_ids)
    # Align prefix dtype to the LLM's token embeddings (Phase 7 T3 fix).
    # Otherwise an fp16 LLM silently upcasts the whole concat to fp32.
    # ``.to(dtype=...)`` is autograd-safe so verbalizer gradients still flow.
    if prefix.dtype != token_embeds.dtype:
        prefix = prefix.to(dtype=token_embeds.dtype)
    inputs_embeds = torch.cat([prefix, token_embeds], dim=1)

    prefix_labels = torch.full(
        (batch_size, num_prefix),
        -100,
        dtype=torch.long,
        device=llm_device,
    )
    labels = torch.cat([prefix_labels, token_ids], dim=1)

    attn_mask = torch.ones(
        batch_size,
        num_prefix + num_tokens,
        dtype=torch.long,
        device=llm_device,
    )

    out = chat_head.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        labels=labels,
    )
    return cast(torch.Tensor, out.loss)


def text_to_state(
    *,
    text: str,
    soma: Any,
    tokenizer: Any,
    encoder: Any,
    soma_output_dim: int,
) -> torch.Tensor:
    """Feed ``text`` through SOMA (no-grad) and return the pooled OUTPUT state.

    Pipeline
    --------
    1. ``encoder.encode(text)`` yields per-token embeddings (the standard
       ``soma.io.text_encoder.TextEncoder`` already composes tokenization +
       token/position embeddings; ``tokenizer`` is accepted as a kwarg so
       callers that plumb a separate tokenizer can still verify agreement,
       but we do NOT re-tokenize here — that would double-count position
       embeddings and de-sync with the encoder's BPE table).
    2. Each embedding is fed as a single-token sensor input to
       ``soma.step(inputs={"text": emb}, eval_mode=True)``. ``eval_mode``
       tells SOMA to skip growth, consolidation, and Hebbian/backprop
       weight updates (read-only pass). The whole loop lives inside a
       ``torch.no_grad()`` block so no autograd graph is built on top of
       SOMA's (frozen-during-bootstrap) learnable parameters.
    3. After the last step, OUTPUT-node activations are read via
       ``soma._current_output_activations()`` and collapsed to a single
       ``(1, soma_output_dim)`` vector by ``SomaAggregator.collapse``.

    Parameters
    ----------
    text:
        UTF-8 input string. Empty text is a legal no-op: no ``soma.step``
        runs, OUTPUT nodes stay unwarmed, and ``SomaAggregator.collapse``
        returns a zero vector.
    soma:
        A ``SOMA`` instance with a text sensor.
    tokenizer:
        The ``tokenizers.Tokenizer`` backing ``encoder`` (kept for API
        symmetry with the caller contract; not used directly here because
        the encoder already owns tokenization).
    encoder:
        A ``TextEncoder`` (or compatible object) exposing ``.encode(text)
        -> list[Tensor]`` where each tensor is ``(sensor_output_dim,)``.
    soma_output_dim:
        The dim each OUTPUT-node activation will have. Must match SOMA's
        actual OUTPUT-node ``output_dim`` (today that equals
        ``config.sensor_output_dim`` — see ``SOMA._initialize_seed_graph``).

    Returns
    -------
    torch.Tensor
        Detached ``(1, soma_output_dim)`` vector; does NOT require grad.
        Callers that need grad flow into SOMA should be building their own
        graph via a trainable adapter on top of this state vector — this
        helper is explicitly the "frozen SOMA" boundary.
    """
    from soma.io.verbalizer import SomaAggregator

    # ``tokenizer`` is accepted per the caller contract (T5 / bootstrap loop
    # wants to pass the same tokenizer it uses downstream for labels) but
    # the TextEncoder already owns tokenization; re-tokenizing here would
    # strip position embeddings and drop any encoder-side truncation.
    del tokenizer

    with torch.no_grad():
        embeddings = encoder.encode(text)  # list[Tensor(embed_dim,)]
        for emb in embeddings:
            soma.step(inputs={"text": emb}, eval_mode=True)

    output_acts = soma._current_output_activations()
    pooled = SomaAggregator.collapse(output_acts, soma_output_dim=soma_output_dim)
    # Defensive: collapse already detaches via view-on-detached inputs, but
    # belt-and-braces — the verbalizer grafts its own autograd graph on top
    # of this vector and we must not chain gradients back into SOMA.
    return pooled.detach()
