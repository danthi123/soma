"""Bootstrap training loop for the SomaVerbalizer.

Only the verbalizer's projector trains. SOMA stays frozen (via
``torch.no_grad`` during state production — enforced in later tasks),
and the ChatHead's LLM stays frozen (via Phase 3's ``requires_grad=False``
contract — enforced at trainer init).
"""

from __future__ import annotations

from typing import Any, cast

import torch

from soma.core.config import SOMAConfig


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
    ) -> None:
        self.soma = soma
        self.verbalizer = verbalizer
        self.chat_head = chat_head
        self.config = config

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

    token_embeds = chat_head.model.get_input_embeddings()(token_ids)
    inputs_embeds = torch.cat([prefix, token_embeds], dim=1)

    prefix_labels = torch.full(
        (batch_size, num_prefix),
        -100,
        dtype=torch.long,
        device=token_ids.device,
    )
    labels = torch.cat([prefix_labels, token_ids], dim=1)

    attn_mask = torch.ones(
        batch_size,
        num_prefix + num_tokens,
        dtype=torch.long,
        device=token_ids.device,
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
