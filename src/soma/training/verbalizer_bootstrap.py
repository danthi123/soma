"""Bootstrap training loop for the SomaVerbalizer.

Only the verbalizer's projector trains. SOMA stays frozen (via
``torch.no_grad`` during state production — enforced in later tasks),
and the ChatHead's LLM stays frozen (via Phase 3's ``requires_grad=False``
contract — enforced at trainer init).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import torch

from soma.core.config import SOMAConfig

_log = logging.getLogger(__name__)


def _cosine_lr(step: int, base_lr: float, max_steps: int, warmup_steps: int) -> float:
    """Linear warmup to ``base_lr`` over ``warmup_steps``, then half-cosine
    decay to zero by ``max_steps``. ``step`` is 1-based."""
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    decay_total = max(max_steps - warmup_steps, 1)
    progress = (step - warmup_steps) / decay_total
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


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

    def _step_loss(self, *, texts: list[str]) -> tuple[torch.Tensor, float]:
        """Forward pass: produce LM loss tensor + its scalar value for a batch.

        Each text goes through SOMA sequentially (SOMA is stateful and can't
        be trivially parallelised), pooled states are stacked into a single
        ``(B, soma_output_dim)`` tensor, and the verbalizer + LLM forward
        run as one batched pass. Per-sample padding is handled via the HF
        tokenizer's ``attention_mask`` so pad tokens are masked out of both
        attention and the CE label positions.

        Callers can pass a single-element list for unbatched behaviour — the
        loss reduces to the same mean-over-valid-tokens scalar a B=1 forward
        would produce (padded positions contribute zero). No ``zero_grad``,
        ``backward``, or ``optim.step`` — caller owns those.
        """
        if not texts:
            raise ValueError("_step_loss requires at least one text")

        states = [
            text_to_state(
                text=t,
                soma=self.soma,
                tokenizer=self._tokenizer,
                encoder=self._encoder,
                soma_output_dim=self.verbalizer.spec.soma_output_dim,
            )
            for t in texts
        ]
        stacked = torch.cat(states, dim=0)  # (B, soma_output_dim)
        prefix = self.verbalizer(stacked)  # (B, k, d_model)

        tok_out = self.chat_head.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
        )
        token_ids = tok_out["input_ids"]
        attn_mask = tok_out.get("attention_mask")

        loss = compute_lm_loss(
            chat_head=self.chat_head,
            prefix=prefix,
            token_ids=token_ids,
            token_attention_mask=attn_mask,
        )
        return loss, float(loss.item())

    def train_step(self, *, text: str) -> float:
        """Single-text convenience wrapper around ``train_batch``.

        Kept as the public training entrypoint for backward-compat; the
        underlying plumbing batches internally via ``_step_loss`` / the
        ``train_batch`` helper below.
        """
        return self.train_batch(texts=[text])

    def train_batch(self, *, texts: list[str]) -> float:
        """One forward + backward + optim.step over a batch of texts.

        Gradient flows only through the verbalizer's two Linear layers.
        SOMA is wrapped in ``torch.no_grad`` inside ``text_to_state`` (T4);
        ChatHead is frozen via the invariant enforced at ``__init__`` (T2).

        Returns the scalar batch loss (mean over valid target tokens).
        Non-finite losses are returned unchanged and skip the optim step
        to protect Adam's running moments — same contract as the previous
        single-text ``train_step``.
        """
        self.verbalizer.train()
        self.optim.zero_grad()

        loss, loss_value = self._step_loss(texts=texts)

        if not math.isfinite(loss_value):
            preview = (texts[0] if texts else "")[:60]
            _log.warning(
                "verbalizer train_batch skipped: non-finite loss %r "
                "(batch_size=%d, first text=%r) — no backward/optim.step.",
                loss_value,
                len(texts),
                preview,
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
        eval_texts: list[str] | None = None,
        eval_interval: int = 1000,
        grad_accum_steps: int = 1,
        cosine_lr: bool = False,
        warmup_steps: int = 0,
        loss_log_path: Path | None = None,
        batch_size: int = 1,
    ) -> list[float]:
        """Run up to ``max_steps`` micro-batches over ``corpus``.

        Default behaviour matches the pre-improvements contract: one sample
        per optim step, constant LR, periodic ``verbalizer_step_N/`` saves
        every ``config.verbalizer_checkpoint_interval`` micro-steps, plus a
        final ``verbalizer_final/``. Returns per-micro-step loss values.

        Parameters
        ----------
        corpus:
            Text iterator. Exits early on ``StopIteration``.
        max_steps:
            Maximum number of micro-batches.
        out_dir:
            Destination directory for verbalizer checkpoints.
        eval_texts:
            If provided, runs ``eval_lm_loss(texts=eval_texts)`` every
            ``eval_interval`` micro-steps. When eval loss improves over the
            prior best, the verbalizer is saved to ``out_dir/verbalizer_best``
            alongside an ``out_dir/best_info.json`` pointing at the winning
            step + loss.
        eval_interval:
            Micro-step period between eval runs (only used if ``eval_texts``
            is non-empty). Default 1000.
        grad_accum_steps:
            Accumulate gradients over this many micro-batches before calling
            ``optim.step``. Per-micro-batch loss is divided by
            ``grad_accum_steps`` so the effective gradient magnitude matches
            a single-pass batch. Default 1 (no accumulation).
        cosine_lr:
            If True, drive the optimiser's LR with a linear warmup to
            ``config.verbalizer_lr`` over ``warmup_steps`` steps, then a
            half-cosine decay to zero by ``max_steps``. Default False
            (constant LR).
        warmup_steps:
            Warmup length used when ``cosine_lr`` is True.
        loss_log_path:
            Optional CSV path. One row per micro-step:
            ``step,train_loss,eval_loss,lr``. ``eval_loss`` is empty on
            non-eval steps. Flushed per row for crash-safety.
        batch_size:
            Number of corpus windows per forward pass. Default 1. Larger
            values amortise the frozen-LLM forward across multiple samples
            (~near-linear throughput gain up to GPU saturation). Variable-
            length windows are padded; pad positions receive ``-100`` label
            masking so they don't contribute to the loss.
        """
        if grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if cosine_lr and warmup_steps > max_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must not exceed max_steps ({max_steps})"
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        out_dir.mkdir(parents=True, exist_ok=True)
        losses: list[float] = []
        corpus_iter = iter(corpus)
        interval = self.config.verbalizer_checkpoint_interval
        base_lr = self.config.verbalizer_lr

        loss_log_fh: Any = None
        if loss_log_path is not None:
            loss_log_path.parent.mkdir(parents=True, exist_ok=True)
            loss_log_fh = loss_log_path.open("w", encoding="utf-8")
            loss_log_fh.write("step,train_loss,eval_loss,lr\n")
            loss_log_fh.flush()

        best_eval_loss = float("inf")
        best_step: int | None = None

        try:
            self.verbalizer.train()
            self.optim.zero_grad()
            accumulated = 0

            corpus_exhausted = False
            for step in range(1, max_steps + 1):
                batch: list[str] = []
                try:
                    for _ in range(batch_size):
                        batch.append(next(corpus_iter))
                except StopIteration:
                    corpus_exhausted = True
                    if not batch:
                        break
                    # Partial batch at the tail of the corpus — still worth a
                    # forward pass, then we exit after the step logic runs.

                # Apply LR schedule first so a NaN micro-batch doesn't pause
                # the cosine decay (it also keeps the CSV log honest).
                if cosine_lr:
                    lr_now = _cosine_lr(step, base_lr, max_steps, warmup_steps)
                    for pg in self.optim.param_groups:
                        pg["lr"] = lr_now
                else:
                    lr_now = base_lr

                loss, loss_value = self._step_loss(texts=batch)

                if math.isfinite(loss_value):
                    (loss / grad_accum_steps).backward()  # type: ignore[no-untyped-call]
                    accumulated += 1
                    if accumulated >= grad_accum_steps:
                        self.optim.step()
                        self.optim.zero_grad()
                        accumulated = 0
                else:
                    # Non-finite loss would corrupt any prior accumulated
                    # grads via the shared autograd buffer, so zero the
                    # bucket and start fresh. Still run eval / log below
                    # so cosine LR + eval cadence keep ticking.
                    _log.warning(
                        "verbalizer train skipped: non-finite loss %r on "
                        "batch_size=%d (first text=%r) — discarding %d "
                        "accumulated micro-batch(es).",
                        loss_value,
                        len(batch),
                        batch[0][:60],
                        accumulated,
                    )
                    self.optim.zero_grad()
                    accumulated = 0

                losses.append(loss_value)

                eval_loss: float | None = None
                if eval_texts and eval_interval > 0 and step % eval_interval == 0:
                    eval_loss = self.eval_lm_loss(texts=eval_texts)
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_step = step
                        self.verbalizer.save(out_dir / "verbalizer_best")
                        (out_dir / "best_info.json").write_text(
                            json.dumps(
                                {"step": step, "eval_loss": round(eval_loss, 6)},
                                indent=2,
                            ),
                            encoding="utf-8",
                        )

                if step % interval == 0:
                    self.verbalizer.save(out_dir / f"verbalizer_step_{step}")

                if loss_log_fh is not None:
                    train_field = f"{loss_value:.6f}" if math.isfinite(loss_value) else "nan"
                    eval_field = f"{eval_loss:.6f}" if eval_loss is not None else ""
                    loss_log_fh.write(f"{step},{train_field},{eval_field},{lr_now:.6g}\n")
                    loss_log_fh.flush()

                if corpus_exhausted:
                    break

            # Flush any leftover grads from a partial accumulation bucket.
            if accumulated > 0:
                self.optim.step()
                self.optim.zero_grad()

            self.verbalizer.save(out_dir / "verbalizer_final")
        finally:
            if loss_log_fh is not None:
                loss_log_fh.close()

        if best_step is not None:
            _log.info(
                "best verbalizer saved at step %d with eval_loss=%.4f",
                best_step,
                best_eval_loss,
            )
        return losses

    @torch.no_grad()
    def eval_lm_loss(self, *, texts: list[str], batch_size: int = 8) -> float:
        """Mean LM loss over held-out texts. No gradient, no optim step.

        Runs in mini-batches of up to ``batch_size`` (default 8) to amortise
        the LLM forward across held-out windows. Per-batch losses are
        weighted by the number of valid target tokens so the final mean
        matches what a single giant forward would compute — padded windows
        don't inflate shorter ones' share of the loss.

        Puts the verbalizer in inference mode (``.train(False)``) for the
        duration, then restores training mode before returning — so callers
        can freely alternate eval and train calls without manual bookkeeping.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.verbalizer.train(False)
        try:
            total_loss_tokens = 0.0
            total_tokens = 0
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                states = [
                    text_to_state(
                        text=t,
                        soma=self.soma,
                        tokenizer=self._tokenizer,
                        encoder=self._encoder,
                        soma_output_dim=self.verbalizer.spec.soma_output_dim,
                    )
                    for t in batch
                ]
                stacked = torch.cat(states, dim=0)
                prefix = self.verbalizer(stacked)
                tok_out = self.chat_head.tokenizer(
                    batch,
                    padding=True,
                    return_tensors="pt",
                )
                attn_mask = tok_out.get("attention_mask")
                loss = compute_lm_loss(
                    chat_head=self.chat_head,
                    prefix=prefix,
                    token_ids=tok_out["input_ids"],
                    token_attention_mask=attn_mask,
                )
                # HF's loss is mean over valid target tokens in the batch;
                # reconstruct the per-batch token weight so the final mean
                # across batches is still token-weighted rather than batch-
                # weighted (matters when batches have very different token
                # counts, e.g. a partial tail batch).
                if attn_mask is not None:
                    num_valid = int(attn_mask.sum().item())
                else:
                    num_valid = int(tok_out["input_ids"].numel())
                if num_valid > 0:
                    total_loss_tokens += float(loss.item()) * num_valid
                    total_tokens += num_valid
        finally:
            self.verbalizer.train(True)
        return total_loss_tokens / total_tokens if total_tokens > 0 else 0.0


def compute_lm_loss(
    *,
    chat_head: Any,
    prefix: torch.Tensor,
    token_ids: torch.Tensor,
    token_attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal-LM cross-entropy conditioned on a continuous prefix.

    ``prefix`` is the (B, k, d_model) soft-prompt from the verbalizer;
    ``token_ids`` is the (B, T) target text. ``token_attention_mask`` is
    an optional (B, T) tensor with 1s on real target tokens and 0s on
    pad tokens — produced by ``tokenizer(texts, padding=True, ...)`` when
    the batch contains variable-length sequences.

    Prefix positions are masked via ``-100`` labels so only real-token
    predictions contribute to the loss; when an attention mask is supplied,
    pad positions also get ``-100`` labels and are excluded from attention.
    Gradient flows through ``prefix`` back into the verbalizer; the LLM
    stays frozen (no grads recorded).
    """
    batch_size, num_prefix, _ = prefix.shape
    _, num_tokens = token_ids.shape

    # Align token_ids to the LLM's device before embedding (HF tokenizers
    # return CPU tensors regardless of where the model lives). Mirrors
    # the Phase 7 T6 fix in ChatSession.respond / SOMA.chat.
    llm_device = chat_head.model.get_input_embeddings().weight.device
    if token_ids.device != llm_device:
        token_ids = token_ids.to(llm_device)
    if token_attention_mask is not None and token_attention_mask.device != llm_device:
        token_attention_mask = token_attention_mask.to(llm_device)

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
    if token_attention_mask is not None:
        token_labels = token_ids.masked_fill(token_attention_mask == 0, -100)
    else:
        token_labels = token_ids
    labels = torch.cat([prefix_labels, token_labels], dim=1)

    prefix_attn = torch.ones(
        batch_size,
        num_prefix,
        dtype=torch.long,
        device=llm_device,
    )
    if token_attention_mask is not None:
        attn_mask = torch.cat([prefix_attn, token_attention_mask.long()], dim=1)
    else:
        token_attn = torch.ones(
            batch_size,
            num_tokens,
            dtype=torch.long,
            device=llm_device,
        )
        attn_mask = torch.cat([prefix_attn, token_attn], dim=1)

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
