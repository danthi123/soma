"""Research direction C: plastic graph under the memory layer.

Sub-phases (see ``docs/plans/2026-04-16-research-c-graph-memory.md``):

* C1 — synthetic-signal harness + first "is there any signal?" run.
  Files: :mod:`research.graph_memory.synthetic_corpus`,
  :mod:`research.graph_memory.harness`,
  :mod:`research.graph_memory.baselines`.
* C1b (contingent) — failure-mode analysis if C1's gate fails.
* C2 — LoCoMo retrieval benchmark with graph blend / expand modes.
* C3 — learned blend, multi-hop, centrality.
* C4 — query-time consolidation pulse.
* C5 — integrate winner into ``soma.memory.api`` as an opt-in mode.

C1's harness is the load-bearing piece. Everything later is gated
on C1 actually showing the graph carries some retrieval signal.
"""
