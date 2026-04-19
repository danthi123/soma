"""SOMA MemoryLayer adapter for the benchmark harness."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from soma.memory import ConversationalMemory, MemoryLayer

from .base import BaseMemorySystem, BenchmarkHit


class SomaAdapter(BaseMemorySystem):
    """Wraps ``soma.memory.MemoryLayer``.

    By default uses sentence-transformers for embeddings (strongest
    baseline; same as Chroma's default). Set ``use_sbert=False`` to fall
    back to SOMA's TextEncoder for fully-offline/tiny footprint runs.

    ``eager_stable_capture`` (default True) pins the pre-Phase-13
    behaviour where ``consolidate()`` implicitly ran the O(N)
    stable-capture pass. The plasticity and graph-ablation benchmarks
    rely on that so their reported ``consolidate`` timing + retrieve
    latencies stay comparable across commits. New benchmarks that
    want to measure the lazy path should pass
    ``eager_stable_capture=False`` — the first retrieve after
    consolidate will then pay the capture cost once, which surfaces
    as retrieve p99 rather than consolidate time.
    """

    name = "soma"

    def __init__(
        self,
        *,
        use_sbert: bool = True,
        attach_soma: bool = False,
        embed_model: str = "all-MiniLM-L6-v2",
        auto_consolidate_every: int = 0,
        graph_rerank_alpha: float = 0.0,
        graph_rerank_stable_capture: bool = True,
        faiss_index_type: str = "flat",
        faiss_threshold: int = 10_000,
        eager_stable_capture: bool = True,
        synap_locality: float = 0.0,
        synap_interval: int | None = None,
        synap_rate: float | None = None,
        seed: int | None = None,
        projection_mode: str | None = None,
        projection_distillation_target: str | None = None,
        projection_distillation_model: str | None = None,
        projection_distillation_weight: float | None = None,
        teacher_cache_dir: str | None = None,
    ) -> None:
        self._use_sbert = use_sbert
        self._attach_soma = attach_soma
        self._embed_model = embed_model
        self._auto_consolidate_every = auto_consolidate_every
        self._graph_rerank_alpha = graph_rerank_alpha
        self._graph_rerank_stable_capture = graph_rerank_stable_capture
        self._faiss_index_type = faiss_index_type
        self._faiss_threshold = faiss_threshold
        self._eager_stable_capture = eager_stable_capture
        # Hard positional-locality cutoff for synaptogenesis. 0.0 (default)
        # disables the filter; 0.5 enables the "locality" mechanism that
        # multi-seed validation on the v0.5 capacity schedule identified
        # as the key design principle for structural plasticity on this
        # substrate (commits 2ba566b, 28c5329). Opt-in here so existing
        # benchmarks reproduce exactly; retrieval/plasticity benchmarks
        # that want to compare locality on/off flip this flag.
        self._synap_locality = synap_locality
        # Override synaptogenesis cadence / rate. None (default) preserves
        # the whitepaper SOMAConfig defaults (interval=100, rate=0.01)
        # which are tuned for 50K-node deployments and barely fire on a
        # 50-fact benchmark workload — making locality-filter ablations
        # vacuous. Plasticity benchmarks that want to exercise synap
        # actively should set interval~=10 and rate~=2.0 (matches
        # SOMAConfig.developmental() tuning).
        self._synap_interval = synap_interval
        self._synap_rate = synap_rate
        # SOMA config seed. None (default) preserves pre-existing
        # nondeterministic behavior; explicit seed is useful for paired
        # comparisons (e.g. locality on/off at matched rng init).
        self._seed = seed
        # Direction 4a: LLM-distilled projections. None defaults preserve
        # pre-Direction-4a behavior. When distillation is active, the
        # adapter will attach a teacher (CachedEmbedder over OllamaEmbedder)
        # to the PredictiveSOMA during prepare(). projection_mode must
        # also be "learnable" for distillation to have any effect.
        self._projection_mode = projection_mode
        self._projection_distillation_target = projection_distillation_target
        self._projection_distillation_model = projection_distillation_model
        self._projection_distillation_weight = projection_distillation_weight
        self._teacher_cache_dir = teacher_cache_dir
        self._mem: MemoryLayer | None = None
        self._bundle_path: Path | None = None

    def prepare(self) -> None:
        if self._use_sbert:
            self._mem = MemoryLayer.with_sbert(self._embed_model)
            self._mem._auto_consolidate_every = self._auto_consolidate_every
            self._mem._graph_rerank_alpha = self._graph_rerank_alpha
            self._mem._graph_rerank_stable_capture = self._graph_rerank_stable_capture
            self._mem._faiss_index_type = self._faiss_index_type
            self._mem._faiss_threshold = self._faiss_threshold
        else:
            from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer

            tokenizer = train_bpe_tokenizer(["placeholder"], vocab_size=128)
            encoder = TextEncoder(tokenizer, embed_dim=32, max_seq_len=128)
            self._mem = MemoryLayer(
                tokenizer=tokenizer,
                encoder=encoder,
                auto_consolidate_every=self._auto_consolidate_every,
                graph_rerank_alpha=self._graph_rerank_alpha,
                graph_rerank_stable_capture=self._graph_rerank_stable_capture,
                faiss_index_type=self._faiss_index_type,
                faiss_threshold=self._faiss_threshold,
            )
        if self._attach_soma:
            from soma.core.config import SOMAConfig
            from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
            from soma.system import SOMA

            config_kwargs: dict[str, object] = dict(
                vocab_size=128,
                text_embed_dim=32,
                sensor_output_dim=32,
                max_input_tokens=128,
                synaptogenesis_max_distance=self._synap_locality,
                seed=self._seed,
            )
            if self._synap_interval is not None:
                config_kwargs["synaptogenesis_interval"] = self._synap_interval
            if self._synap_rate is not None:
                config_kwargs["synaptogenesis_rate"] = self._synap_rate
            if self._projection_mode is not None:
                config_kwargs["projection_mode"] = self._projection_mode
            if self._projection_distillation_target is not None:
                config_kwargs["projection_distillation_target"] = (
                    self._projection_distillation_target
                )
            if self._projection_distillation_model is not None:
                config_kwargs["projection_distillation_model"] = (
                    self._projection_distillation_model
                )
            if self._projection_distillation_weight is not None:
                config_kwargs["projection_distillation_weight"] = (
                    self._projection_distillation_weight
                )
            # Use developmental() as the base when active-growth overrides
            # are requested so tuned helpers like activation_threshold=0.005
            # (vs whitepaper default 0.1) kick in. Without this, a 32-dim
            # substrate rarely clears the activation threshold and synap
            # never fires even at interval=10, rate=2.0.
            if self._synap_interval is not None or self._synap_rate is not None:
                config = SOMAConfig.developmental(**config_kwargs)
            else:
                config = SOMAConfig(**config_kwargs)
            soma = SOMA(config)
            # SOMA's graph operates on its own small TextEncoder regardless
            # of what the MemoryLayer embeds with for cosine. Keeping them
            # independent lets sbert (384-d) drive retrieval while the 32-d
            # SOMA substrate drives graph-based re-ranking.
            if self._use_sbert:
                soma_tokenizer = train_bpe_tokenizer(["placeholder"], vocab_size=128)
                soma_encoder = TextEncoder(
                    soma_tokenizer, embed_dim=32, max_seq_len=128,
                )
            else:
                assert self._mem._encoder is not None
                soma_tokenizer = self._mem._tokenizer
                soma_encoder = self._mem._encoder
            self._mem.attach_soma(soma, soma_tokenizer, soma_encoder)

    def store(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        assert self._mem is not None
        return self._mem.store(text, metadata=metadata)

    def store_with_embedding(
        self,
        text: str,
        embedding: Any,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Skip the per-call embed by injecting the precomputed vector
        through MemoryLayer's internals — we set the custom embed_fn to
        a closure that returns this exact vector for this exact text."""
        assert self._mem is not None
        import torch

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding)
        # The MemoryLayer's _embed() routes through _custom_embed_fn when
        # set. Temporarily override for this single store; the sbert
        # path remains the default for retrieve.
        prev_fn = self._mem._custom_embed_fn
        self._mem._custom_embed_fn = lambda _t: embedding
        try:
            return self._mem.store(text, metadata=metadata)
        finally:
            self._mem._custom_embed_fn = prev_fn

    def retrieve(self, query: str, k: int = 5) -> list[BenchmarkHit]:
        assert self._mem is not None
        hits = self._mem.retrieve(query, k=k)
        return [
            BenchmarkHit(
                text=h.text, score=h.score, metadata=h.metadata, node_id=h.node_id,
            )
            for h in hits
        ]

    def consolidate(self) -> None:
        assert self._mem is not None
        self._mem.consolidate()
        # Phase 13: consolidate() now defers the O(N) stable-capture
        # pass to the first retrieve. Benchmarks that measure the
        # "consolidate" wall clock as the total amortized cost of
        # writing expect the eager pre-Phase-13 behaviour — run the
        # capture here so the number stays comparable to older
        # reports. Set ``eager_stable_capture=False`` to pin the new
        # lazy path instead (capture cost surfaces as retrieve p99).
        if self._eager_stable_capture:
            self._mem.stable_capture()

    def clear(self) -> None:
        self._mem = None
        if self._bundle_path and self._bundle_path.exists():
            shutil.rmtree(self._bundle_path, ignore_errors=True)
            self._bundle_path = None

    def disk_footprint_bytes(self) -> int:
        if self._mem is None:
            return 0
        self._bundle_path = Path(tempfile.mkdtemp()) / "mem"
        self._mem.save(self._bundle_path)
        return sum(
            f.stat().st_size for f in self._bundle_path.rglob("*") if f.is_file()
        )

    def teardown(self) -> None:
        self.clear()


class ConversationalSomaAdapter(SomaAdapter):
    """SomaAdapter + ConversationalMemory wrapper for the LoCoMo benchmark.

    Same interface as :class:`SomaAdapter` so the harness runner can
    swap one for the other via ``run_locomo.py --conversational``.
    ``store(text)`` routes through ``ConversationalMemory.add_message``
    (with a ``role="user"`` default) so every turn triggers fact
    extraction and reconciliation. ``retrieve(query)`` goes through
    ``ConversationalMemory.retrieve`` so superseded entries are
    filtered out by default.

    Optional ``extractor_llm=`` kwarg is forwarded to
    :class:`ConversationalMemory` so benchmark users can pin a stronger
    JSON-reliable model for extract + reconcile while keeping a smaller
    one for chat + summary.

    Reports the per-run "facts stored / turns processed" ratio so the
    report can surface how much structure the LLM pulled out of raw
    turns.
    """

    name = "soma-conversational"

    def __init__(
        self,
        *,
        llm: Any,
        extractor_llm: Any = None,
        session_id: str | None = None,
        summary_every: int = 20,
        near_dup_threshold: float = 0.92,
        ambiguous_threshold: float = 0.75,
        embed_fn: Any = None,
        embed_dim: int | None = None,
        **soma_kwargs: Any,
    ) -> None:
        # embed_fn path is used in tests; production runs go through
        # sbert via the parent's prepare().
        use_sbert = embed_fn is None
        super().__init__(use_sbert=use_sbert, **soma_kwargs)
        self._llm = llm
        # Optional stronger model for the structured-JSON steps
        # (extract + reconcile). None = fall back to the main llm.
        self._extractor_llm = extractor_llm
        self._cm_session_id = session_id or "locomo"
        self._summary_every = summary_every
        self._near_dup_threshold = near_dup_threshold
        self._ambiguous_threshold = ambiguous_threshold
        self._embed_fn = embed_fn
        self._embed_dim = embed_dim
        self._cm: ConversationalMemory | None = None
        self.facts_stored: int = 0
        self.turns_processed: int = 0

    def prepare(self) -> None:
        if self._embed_fn is not None:
            assert self._embed_dim is not None
            self._mem = MemoryLayer(
                embed_fn=self._embed_fn, embed_dim=self._embed_dim,
            )
        else:
            super().prepare()
        assert self._mem is not None
        self._cm = ConversationalMemory(
            memory=self._mem,
            llm=self._llm,
            extractor_llm=self._extractor_llm,
            session_id=self._cm_session_id,
            summary_every=self._summary_every,
            near_dup_threshold=self._near_dup_threshold,
            ambiguous_threshold=self._ambiguous_threshold,
        )
        self.facts_stored = 0
        self.turns_processed = 0

    def store(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Route turns through ConversationalMemory so extract + reconcile fire.

        Returns a stable id — we reuse the raw turn's node_id for the
        harness's evidence-matching (the turn is still stored verbatim;
        the fact/summary extras are additive).
        """
        assert self._cm is not None
        assert self._mem is not None
        # Pre-count facts so we can attribute new ones to this turn.
        before_facts = sum(
            1 for m in self._mem._metadatas if m.get("type") == "fact"
        )
        # Speaker defaults to "user" so fact extraction always fires on
        # LoCoMo turns; the benchmark's speaker field is in metadata.
        role = "user"
        self._cm.add_message(role, text, metadata=metadata or {})
        after_facts = sum(
            1 for m in self._mem._metadatas if m.get("type") == "fact"
        )
        self.facts_stored += max(0, after_facts - before_facts)
        self.turns_processed += 1
        # Return the raw turn's id (last entry with type=turn for this session).
        for nid in reversed(self._mem._ids):
            meta = self._mem.get(nid).metadata if self._mem.get(nid) else {}
            if meta.get("type") == "turn":
                return nid
        return ""

    def retrieve(self, query: str, k: int = 5) -> list[BenchmarkHit]:
        assert self._cm is not None
        hits = self._cm.retrieve(query, k=k)
        return [
            BenchmarkHit(
                text=h.text,
                score=h.score,
                metadata=h.metadata,
                node_id=h.node_id,
            )
            for h in hits
        ]

    def clear(self) -> None:
        self._cm = None
        super().clear()

    def teardown(self) -> None:
        self.clear()
