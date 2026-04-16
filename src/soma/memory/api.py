"""Public memory-layer API: vector-DB-shaped surface over SOMA's substrate.

This is the product-facing entry point introduced by the 2026-04-15
pivot (see ``docs/plans/2026-04-15-memory-layer-pivot.md``). It gives
agent developers a familiar ``store``/``retrieve`` contract while
leaving room for the graph/plasticity differentiators to come online in
later stages.

Stage 2 (this module) ships the flat vector-store semantics plus
save/load. Stage 3 wires ``consolidate()`` into SOMA's growth engine so
stored entries become graph structure that prunes and reinforces with
use; today ``consolidate`` is a safe no-op so callers can include the
call in their loops now and not have to revisit when the plasticity
path lands.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F  # noqa: N812

from soma.io.text_encoder import TextEncoder, load_tokenizer

EmbedFn = Callable[[str], torch.Tensor]


@dataclass(frozen=True)
class MemoryHit:
    """One retrieved entry. Immutable so callers can pass them around safely."""

    node_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp_step: int = 0


class MemoryLayer:
    """Local-first, learning agent-memory layer.

    Reads like a vector DB at the edge: ``store(text)`` appends,
    ``retrieve(query)`` ranks by cosine similarity. Behind the API,
    entries are kept alongside their pooled-token-embedding vectors in
    an in-memory tensor that scales linearly with the store size (fine
    for up to ~100K entries on consumer hardware; Stage 3 adds a
    chunked / on-disk path for larger stores).

    The embedder is caller-supplied — pass any
    :class:`soma.io.text_encoder.TextEncoder` plus its tokenizer. This
    keeps the memory layer independent of any specific embedding model,
    so callers can swap in sentence-transformers or an LLM's input
    embeddings once Stage 3's benchmark harness picks a default.

    Persistence writes a directory bundle compatible with
    ``SOMA.save_bundle`` naming: ``tokenizer.json``, ``encoder.pt``,
    ``memory_index.json``, ``memory_embeddings.pt``. A future
    MemoryLayer bundle CAN be dropped inside a SOMA brain bundle and
    the two will coexist without collision.
    """

    def __init__(
        self,
        *,
        tokenizer: Any = None,
        encoder: TextEncoder | None = None,
        embed_fn: EmbedFn | None = None,
        embed_dim: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if embed_fn is None and encoder is None:
            raise ValueError(
                "MemoryLayer needs either (tokenizer + encoder) or embed_fn"
            )
        self._tokenizer = tokenizer
        self._encoder = encoder
        self._custom_embed_fn = embed_fn
        if device is not None:
            self._device = torch.device(device)
        elif encoder is not None:
            self._device = encoder.embedding.weight.device
        else:
            self._device = torch.device("cpu")
        if embed_fn is not None:
            if embed_dim is None:
                raise ValueError("embed_dim required when using embed_fn")
            self._embed_dim: int = embed_dim
        else:
            assert encoder is not None
            self._embed_dim = int(encoder.embed_dim)

        # Optional SOMA attachment for graph-based consolidation.
        self._soma: Any = None
        self._soma_tokenizer: Any = None
        self._soma_encoder: Any = None

        # Parallel storage. Order is preserved across save/load so
        # ``get_recent`` stays stable.
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._timestamps: list[int] = []
        # (N, embed_dim) — lazily rebuilt from _embeddings_list when persisting
        # so we don't pay stack cost on every store.
        self._embeddings_list: list[torch.Tensor] = []

        self._step: int = 0

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def with_sbert(
        cls,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        device: torch.device | str | None = None,
    ) -> MemoryLayer:
        """Create a MemoryLayer backed by a sentence-transformers model.

        Requires ``sentence-transformers`` to be installed (optional dep).
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "MemoryLayer.with_sbert() requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from exc

        model = SentenceTransformer(model_name)
        dim = int(model.get_sentence_embedding_dimension())

        def _embed(text: str) -> torch.Tensor:
            return torch.tensor(model.encode(text, convert_to_numpy=True))

        return cls(embed_fn=_embed, embed_dim=dim, device=device)

    # ------------------------------------------------------------------
    # SOMA graph attachment (optional)
    # ------------------------------------------------------------------
    def attach_soma(
        self,
        soma: Any,
        tokenizer: Any,
        encoder: Any,
    ) -> None:
        """Attach a SOMA instance for graph-based consolidation.

        When attached, :meth:`consolidate` feeds stored entries through
        the SOMA graph, triggering structural plasticity (edge
        formation, pruning, myelination). Without attachment,
        ``consolidate`` remains a safe no-op.
        """
        self._soma = soma
        self._soma_tokenizer = tokenizer
        self._soma_encoder = encoder

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def store(self, text: str, *, metadata: dict[str, Any] | None = None) -> str:
        """Add an entry. Returns a stable node_id (uuid4 hex)."""
        if not text or not text.strip():
            raise ValueError("MemoryLayer.store rejects empty text")
        node_id = uuid.uuid4().hex
        embedding = self._embed(text)
        self._ids.append(node_id)
        self._texts.append(text)
        self._metadatas.append(dict(metadata) if metadata else {})
        self._timestamps.append(self._step)
        self._embeddings_list.append(embedding)
        self._step += 1
        return node_id

    def retrieve(self, query: str, k: int = 5) -> list[MemoryHit]:
        """Return up to k entries most similar to ``query`` by cosine."""
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not self._ids:
            return []
        q_vec = self._embed(query)
        return self._rank(q_vec, k=k, exclude_idx=None)

    def related(self, node_id: str, k: int = 5) -> list[MemoryHit]:
        """Return up to k entries most similar to the entry at ``node_id``."""
        if node_id not in self._ids:
            raise KeyError(f"node_id {node_id!r} not found in MemoryLayer")
        idx = self._ids.index(node_id)
        q_vec = self._embeddings_list[idx]
        return self._rank(q_vec, k=k, exclude_idx=idx)

    def get(self, node_id: str) -> MemoryHit | None:
        """Fetch an entry by id; ``None`` if unknown. Score is self-cosine (1.0)."""
        if node_id not in self._ids:
            return None
        idx = self._ids.index(node_id)
        return MemoryHit(
            node_id=node_id,
            text=self._texts[idx],
            score=1.0,
            metadata=dict(self._metadatas[idx]),
            timestamp_step=self._timestamps[idx],
        )

    def get_recent(self, n: int) -> list[MemoryHit]:
        """Return the n most recently stored entries, newest first."""
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        start = max(0, len(self._ids) - n)
        recent_indices = list(range(start, len(self._ids)))[::-1]
        return [self._hit_for_index(i, score=1.0) for i in recent_indices]

    def forget(self, node_id: str) -> bool:
        """Remove an entry. Returns True if removed, False if unknown."""
        if node_id not in self._ids:
            return False
        idx = self._ids.index(node_id)
        self._ids.pop(idx)
        self._texts.pop(idx)
        self._metadatas.pop(idx)
        self._timestamps.pop(idx)
        self._embeddings_list.pop(idx)
        return True

    def consolidate(self) -> int:
        """Push stored entries through SOMA's graph to trigger plasticity.

        When a SOMA instance is attached via :meth:`attach_soma`, this
        feeds each stored text through the graph (one ``step()`` per
        entry) so structural plasticity — synaptogenesis, pruning,
        myelination — fires based on the content patterns. Returns
        the number of entries processed.

        Without an attached SOMA, this is a safe no-op (returns 0).
        Callers should include ``consolidate()`` in their loops now; it
        becomes load-bearing once a SOMA is attached.
        """
        if self._soma is None:
            return 0
        from soma.training.verbalizer_bootstrap import text_to_state

        soma_output_dim = int(self._soma.config.sensor_output_dim)
        processed = 0
        for text in self._texts:
            text_to_state(
                text=text,
                soma=self._soma,
                tokenizer=self._soma_tokenizer,
                encoder=self._soma_encoder,
                soma_output_dim=soma_output_dim,
            )
            processed += 1
        return processed

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._ids

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Write a self-contained memory bundle to ``path`` (a directory)."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        has_encoder = self._encoder is not None
        if has_encoder:
            self._encoder.save_tokenizer(out / "tokenizer.json")
            torch.save(self._encoder.state_dict(), out / "encoder.pt")
        if self._embeddings_list:
            stacked = torch.stack(self._embeddings_list, dim=0).detach().cpu()
        else:
            stacked = torch.empty((0, self._embed_dim))
        torch.save(stacked, out / "memory_embeddings.pt")
        index: dict[str, Any] = {
            "schema_version": 1,
            "embed_dim": self._embed_dim,
            "embed_type": "text_encoder" if has_encoder else "custom",
            "step": self._step,
            "entries": [
                {
                    "node_id": nid,
                    "text": txt,
                    "metadata": md,
                    "timestamp_step": ts,
                }
                for nid, txt, md, ts in zip(
                    self._ids, self._texts, self._metadatas, self._timestamps,
                    strict=True,
                )
            ],
        }
        if has_encoder:
            index["max_seq_len"] = int(self._encoder.max_seq_len)
        (out / "memory_index.json").write_text(
            json.dumps(index, indent=2), encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        embed_fn: EmbedFn | None = None,
        device: torch.device | str | None = None,
    ) -> MemoryLayer:
        """Rehydrate a MemoryLayer from a ``save()``-produced directory.

        Bundles saved with the TextEncoder path include ``tokenizer.json``
        and ``encoder.pt``; those are reloaded automatically. Bundles saved
        with a custom ``embed_fn`` only store embeddings + index — pass the
        same ``embed_fn`` at load time so new stores can be embedded.
        """
        src = Path(path)
        index_path = src / "memory_index.json"
        embeddings_path = src / "memory_embeddings.pt"
        for required in (index_path, embeddings_path):
            if not required.exists():
                raise FileNotFoundError(f"MemoryLayer bundle missing {required.name}")

        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported MemoryLayer schema version {index.get('schema_version')!r}"
            )

        embed_type = index.get("embed_type", "text_encoder")
        embed_dim = int(index["embed_dim"])

        if embed_type == "text_encoder":
            tokenizer_path = src / "tokenizer.json"
            encoder_path = src / "encoder.pt"
            for f in (tokenizer_path, encoder_path):
                if not f.exists():
                    raise FileNotFoundError(f"MemoryLayer bundle missing {f.name}")
            tokenizer = load_tokenizer(tokenizer_path)
            encoder = TextEncoder(
                tokenizer,
                embed_dim=embed_dim,
                max_seq_len=int(index.get("max_seq_len", 512)),
                device=device,
            )
            encoder_state = torch.load(
                encoder_path, map_location=device or "cpu", weights_only=True,
            )
            encoder.load_state_dict(encoder_state)
            instance = cls(tokenizer=tokenizer, encoder=encoder, device=device)
        else:
            if embed_fn is None:
                raise ValueError(
                    "This bundle was saved with a custom embed_fn; pass the "
                    "same embed_fn to load()."
                )
            instance = cls(
                embed_fn=embed_fn, embed_dim=embed_dim, device=device,
            )
        instance._step = int(index.get("step", 0))
        embeddings = torch.load(embeddings_path, map_location=device or "cpu", weights_only=True)
        target_device = instance._device
        for entry, vec in zip(index["entries"], embeddings, strict=True):
            instance._ids.append(str(entry["node_id"]))
            instance._texts.append(str(entry["text"]))
            instance._metadatas.append(dict(entry.get("metadata", {})))
            instance._timestamps.append(int(entry.get("timestamp_step", 0)))
            instance._embeddings_list.append(vec.to(target_device))
        return instance

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _embed(self, text: str) -> torch.Tensor:
        """Embed ``text`` into a (embed_dim,) vector."""
        if self._custom_embed_fn is not None:
            vec = self._custom_embed_fn(text)
            return vec.detach().to(self._device)
        assert self._encoder is not None
        with torch.no_grad():
            stacked = self._encoder.encode_batch(text)  # (T, embed_dim)
        if stacked.numel() == 0:
            return torch.zeros(self._embed_dim, device=self._device)
        pooled = stacked.mean(dim=0)
        return pooled.detach().to(self._device)

    def _rank(
        self,
        query_vec: torch.Tensor,
        *,
        k: int,
        exclude_idx: int | None,
    ) -> list[MemoryHit]:
        matrix = torch.stack(self._embeddings_list, dim=0)
        sims = F.cosine_similarity(query_vec.unsqueeze(0), matrix, dim=-1)
        if exclude_idx is not None:
            sims = sims.clone()
            sims[exclude_idx] = float("-inf")
        eligible = (sims > float("-inf")).sum().item()
        k = int(min(k, eligible))
        if k <= 0:
            return []
        top = torch.topk(sims, k=k)
        return [
            self._hit_for_index(int(i.item()), score=float(s.item()))
            for i, s in zip(top.indices, top.values, strict=True)
        ]

    def _hit_for_index(self, idx: int, *, score: float) -> MemoryHit:
        return MemoryHit(
            node_id=self._ids[idx],
            text=self._texts[idx],
            score=score,
            metadata=dict(self._metadatas[idx]),
            timestamp_step=self._timestamps[idx],
        )
