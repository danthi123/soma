# Paper Draft Audit

**Date:** 2026-04-18
**Auditor:** Claude (self-review, fact-checked against repo + prior session)
**Target:** `docs/paper/draft.md` as of commit `0390590`

## Status — revision pass complete 2026-04-18

Most audit items have been addressed in a follow-up commit.
Remaining deferred-to-pre-submission items: **D8** (full citations),
**E1** (figures), **E3** (structure-control experiment — needs a
new run), **E5** (Phase 4b FT-curve figure — needs re-running with
logging).

Key corrections discovered during the revision pass:

- **"+275%"** = F1 growth (0.0595 → 0.2235 relative), not graph
  growth. Verified via session-transcript inspection
  (`22e307d9-4c5f-4e1c-bdcf-2d1492d48e17.jsonl`).
- **Cross-session recall** was never directly measured in the
  `6a0a822` experiment. The session-3 F1 is on session-3's own QA.
  Paper claim now scoped accordingly.
- **Phase 4b catastrophic FT**: full numbers are VecDB 63→50 (−13),
  gated 65→51 (−14), gate usage 32→11, after 316 updates. Paper
  now reports all three.
- **Consolidation +45%** context: ad-hoc CLI test, not scripted
  repeated experiment; flagged as evidence-of-life with caveats in
  §8 Limitations.
- **"~8000 fingerprint patterns"** was a back-of-envelope
  derivation, not empirical. Replaced with the exact C(N, 3) count.
- **F1 noise floor** ~±0.05 between checkpoints (from earlier
  work) — added to Limitations as context for the shuffle-diagnostic
  significance.

This document enumerates every issue I found in the draft, classified
by severity. Items are independently addressable — tackle in any order.

---

## A. Factual errors (must fix)

### A1. Table 3 "Delta" column mislabeled
**Location:** §4.3, Table 3, draft lines 285-289.
**Issue:** Column labeled "Delta" but values are Net (W−L). Actual hits−VecDB deltas are +1, −2, 0 (not +8, +4, −1).
**Fix:** Either rename column "Net" and keep +8/+4/−1, or keep "Delta" label and change values to +1/−2/0. Recommend adding BOTH: separate "Delta" and "Net" columns.

### A2. LoCoMo total QA pair count
**Location:** §3.2, line 208.
**Issue:** "1980 QA pairs" — actual is 1986.
**Fix:** "1986 QA pairs".

### A3. "Per-edge input projections" is wrong
**Location:** §5, line 483.
**Issue:** Projections are per-associator-node (`self._input_projections: dict[str, Tensor]` keyed by node id in `prediction.py:53-64`), not per-edge.
**Fix:** "SOMA's per-associator-node input projections are randomly initialized".

### A4. "C(14, 3) ≈ 364 patterns" — applicable-node count off
**Location:** §5, line 491.
**Issue:** Lateral inhibition excludes SENSOR and OUTPUT, so applicable nodes for the 3-winner selection are associators + integrators (12, not 14). C(12, 3) = 220, not 364.
**Fix:** "roughly C(12, 3) = 220 distinct 3-winner sets possible with 8 associators + 4 integrators".

### A5. "+275% graph growth" claim is unverified
**Location:** §4.7 and Abstract.
**Issue:** Commit `6a0a822` body shows: session 1 (40 memories, 29 edges) → session 3 (120 memories, 43 edges). Memory growth is 3x (200% increase or 300% of initial); edge growth is 48%. "275%" doesn't match any computable quantity from the commit body.
**Fix:** Either track down the actual 275% metric or rephrase as "3x memory growth (40 → 120) with preserved cross-session recall" and back the "preserved recall" claim (see C2).

### A6. Experiment count inconsistent (14 vs 16)
**Location:** Abstract ("14 diagnostic experiments"), §9 ("sixteen").
**Issue:** Retrieval phases: 3, 4, 4b, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 = **13**. Env experiments: v0, v0_ablation, v0.5 = **3**. Total = **16**.
**Fix:** Update abstract: "Across 13 retrieval-diagnostic experiments on LoCoMo and LongMemEval, and 3 sequence-prediction experiments".

---

## B. Inconsistencies (must fix)

### B1. "Capacity-matched online MLP" is overclaimed
**Location:** §4.7, line 395 and line 405.
**Issue:** OnlineMLP is 16→64→64→16 ≈ 5600 params. SOMA (n=8) has 14 nodes each with MLP structure, plus integrators, plus graph weights — likely 15-25k params depending on exact counts. Not capacity-matched.
**Fix:** "We compare against an online MLP baseline (2 hidden layers, 64 units; ~5.6k parameters). SOMA has substantially more parameters; this comparison does not attempt parameter-matching but establishes that a standard MLP baseline cannot match SOMA's adaptation curve."

### B2. Selective presentation of adaptation windows
**Location:** §4.7, line 407.
**Issue:** Only first boundary (rw→rot, SOMA=0 vs OnlineMLP=80) is mentioned. Actual three-boundary data:
- rw → rot: SOMA 0, OnlineMLP 80, FrozenMLP 500
- rot → sqrt: SOMA 118, OnlineMLP 28, FrozenMLP 0
- sqrt → mlp: SOMA 500, OnlineMLP 500, FrozenMLP 1

SOMA is **slower** on rot→sqrt (118 vs 28). Both hit the 500-step cap on sqrt→mlp.
**Fix:** Either (i) present full table and note the metric is biased (1.2x of a tiny pre-boundary MSE is a much tighter target), or (ii) explicitly scope the claim: "at the first boundary, where a well-tracking learner's 1.2x-recovery target is achievable from brief exposure, SOMA recovers in 0 steps vs OnlineMLP 80. On later boundaries the metric becomes less informative: SOMA's pre-boundary MSE (~0.001) makes its 1.2x target ~0.0012, an extremely tight bar; OnlineMLP's pre-boundary MSE (~0.045) makes its target 0.054, trivial to hit."

### B3. Abstract vs section summaries claim different positive results
**Location:** Abstract lines 27-32, §4.7 lines 371-467.
**Issue:** Abstract mentions "consolidation +45%" and "multi-session +275% growth" but not the env experiments. §9 conclusion heavily features env experiments. Env is arguably the most surprising finding; abstract doesn't preview it.
**Fix:** Rewrite abstract to briefly mention the two env findings (SOMA 3-40x > MLP on adaptation; plasticity ablation shows no_growth wins 7/8).

### B4. Phase 8 referenced but never introduced
**Location:** §5 line 484 ("Phase 8 learnable-projection experiment").
**Issue:** Reader doesn't know what Phase 8 is. Results section doesn't discuss it separately.
**Fix:** Either (i) add a brief §4.8 describing the Phase 8 learnable-projection experiment and its ~+1 at-noise result, or (ii) drop the cite and just say "we tested a variant that learns the random projections via competitive-learning updates; gains were at the noise floor — see commit `309caeb`."

---

## C. Overclaims (should soften)

### C1. "Load-bearing contribution is the executable graph substrate, specifically: wave-based execution, residual connections, homeostatic gain"
**Location:** §4.7 line 454, §9 line 691.
**Issue:** We have evidence that plasticity is NOT load-bearing and SOMA still beats MLP. We have NOT directly tested which of (wave execution / residuals / homeostasis / depth / parameter count) drives the advantage. Attributing to specific components is unsupported.
**Fix:** "We do not isolate which aspect of SOMA's substrate drives the advantage. The ablation rules out structural plasticity but does not separately test wave execution, residual connections, homeostatic gain, depth, or parameter count. These remain open questions; see §8 Limitations."

### C2. "Preserved cross-session recall"
**Location:** §4.7 line 386, Abstract.
**Issue:** Commit `6a0a822` shows session 3 F1=0.224 improving over session 2 F1=0.015, but "cross-session recall" means "can session 3 retrieve session 1 info?" — which is not explicitly measured in the commit data. Session 3's F1 is on session 3's questions, not session 1's.
**Fix:** Either (a) run the actual cross-session recall test and cite those numbers, or (b) rephrase: "three-session development produced a 3x increase in stored memories with session-3 F1 higher than session-1 F1, suggesting accumulated structure rather than overwriting."

### C3. "Random-projection-driven graphs — the default in most brain-inspired architectures — do not satisfy this condition by construction"
**Location:** §6.2, line 543-545.
**Issue:** We haven't surveyed "most brain-inspired architectures" for their projection initialization. The claim is plausible but unsupported.
**Fix:** Scope it: "Random-projection-driven graphs — which SOMA uses and which are common in brain-inspired architectures — do not satisfy this condition by construction; the structural signal is decoupled from semantic content." Drop the "most ... do not satisfy" framing.

### C4. "Scaling tests disambiguate vocabulary bottleneck from signal quality bottleneck"
**Location:** §6.1, line 533-534.
**Issue:** This is true in our case because both outcomes are plausible ex ante. But the general claim "scaling tests always disambiguate these" is too strong.
**Fix:** "Our scaling sweep disambiguated the two for SOMA" — make it specific.

### C5. "Temporal queries are the largest single loss source"
**Location:** §4.4, line 321.
**Issue:** The delta column for temporal shows +1 (15W/17L). It's NOT a loss source in aggregate. The absolute loss COUNT is 17, which is the largest, but the W/L ratio is near parity.
**Fix:** "Temporal queries are where the graph flips the most rankings (15W + 17L = 32 rank changes vs other's 10W + 7L = 17 changes) — the mechanism is noisy on temporal queries but near-breakeven in delta terms."

---

## D. Gaps (should fill)

### D1. Method section: encoder is not fine-tuned in main experiments
**Location:** §3.
**Gap:** Never explicitly states the encoder is frozen pretrained weights.
**Fill:** Add to §3.2: "The encoder is not fine-tuned in any main-result experiment; it is loaded pretrained and frozen. (Phase 4 and 4b experiments that attempted encoder fine-tuning are discussed in §5 as evidence of the 'semantically arbitrary' claim.)"

### D2. PredictiveSOMA text flow not documented
**Location:** §2.3.
**Gap:** Paper says "encoder → sensor node" but doesn't explain: text → BPE tokens → embedding via pretrained encoder → sensor node input → graph step → prediction head → next-input prediction.
**Fill:** Add diagram or 3-sentence flow to §2.3.

### D3. §4 missing explicit Phase 4b narrative
**Location:** §5 references Phase 4b without §4 having introduced it.
**Fill:** Either add §4.8 "Contrastive encoder fine-tuning (Phase 4b)" describing the experiment and result, or explicitly link back: "The Phase 4b contrastive-fine-tuning experiment (not tabulated in §4; see commit `30cde22`) ..."

### D4. Appendix A mixes retrieval and env configs
**Location:** Appendix A, line 703-717.
**Gap:** Current config shown (`sensor_output_dim=384`) is the retrieval config (384 matches encoder dim). But env experiments used dim=16. Reader doesn't know.
**Fill:** Split Appendix A into A.1 retrieval config and A.2 env config.

### D5. No mention of LongMemEval being "oracle variant"
**Location:** Abstract, §4.6.
**Gap:** Abstract says "LongMemEval (3094-turn corpus, 100 queries)". But LongMemEval has oracle/small/medium variants and 500 total items. We use first 100 of oracle.
**Fill:** Specify "LongMemEval-oracle (first 100 items)".

### D6. Shuffle seeds count is arbitrary
**Location:** §4.2, Table 2.
**Gap:** "5 random permutations" — why 5? Too few to establish statistical significance.
**Fill:** Add to §8 Limitations: "Shuffle diagnostic used 5 permutations, sufficient to show real > max-shuffled but statistically weak. A 100-seed version would quantify the z-score of the real result against the shuffle distribution."

### D7. "Consolidation +45%" context missing
**Location:** §4.7, line 376-380.
**Gap:** Claim is "+45% QA on synthetic" but paper doesn't describe the synthetic corpus, eval protocol, or baseline. Reader can't evaluate how strong this result is.
**Fill:** 3-5 sentence block describing: synthetic corpus shape, consolidation schedule (every N steps), eval method, baseline (non-consolidation or early-stop?), and the 0.279 → 0.404 F1 numbers.

### D8. Related work lacks arxiv IDs / actual citations
**Location:** §7.
**Gap:** SYNAPSE, Graphiti, MAGMA, Mem0, Letta are mentioned but have no citations. For a real submission, these need proper references.
**Fill:** Needs to be done before any submission — compile bibtex entries or arxiv IDs. Note: treat as a future-work item, not a same-day fix.

---

## E. Missing content (would strengthen)

### E1. No figures
**Impact:** Major. A paper without figures is hard to read.
**Suggested figures:**
- **Fig 1 (schematic):** SOMA architecture sketch — sensor → associator graph with lateral inhibition → winner nodes → fingerprint/topology.
- **Fig 2:** MSE-over-time curves for SOMA vs OnlineMLP vs FrozenMLP on the v0 env. Show regime boundaries as vertical lines.
- **Fig 3:** Node-count trajectory for full vs no_growth on v0.5 — visualizes that growth happens but doesn't help.
- **Fig 4:** Shuffle histogram — real delta overlaid on shuffled-delta distribution.

I have the raw JSON data to generate all four. Matplotlib is already a dep.

### E2. No "negative-result methodology" framing
**Impact:** Paper could be positioned as a negative-result methodology template, but current framing is "SOMA empirical study." A small framing change in the intro could broaden appeal.
**Suggested edit:** Add one paragraph to §1 introduction: "Our broader aim is to offer a **diagnostic template** for brain-inspired retrieval claims, comprising multi-slice held-out, shuffle, scaling, and cross-benchmark tests. We apply this template to SOMA; the same template is directly applicable to any graph-augmented retrieval system."

### E3. No ablation-vs-structure-control comparison
**Impact:** See C1 — we don't know if the advantage is from wave execution, residual nodes, or just more params. A "structure-matched" feedforward MLP (4-6 hidden layers, same total parameters as SOMA) would test this directly.
**Cost:** ~30 min to code + run.
**Recommendation:** Add this as a follow-up experiment before paper submission.

### E4. No "why tanh?" explanation for env
**Impact:** Minor. The env applies tanh to keep observations bounded (to avoid the first-attempt NaN blow-up). A reader might wonder.
**Fill:** One-liner: "All regimes apply a final tanh to bound observations in [-1, 1]^D, keeping the observation space stationary across regime shifts."

### E5. Figure from Phase 4b destruction
**Impact:** The Phase 4b result (encoder destruction via contrastive FT) is central to the §5 argument but presented only as "-13 hits on 100 queries." A simple before/after F1 curve over FT updates would be compelling visual evidence.
**Cost:** Would require re-running the Phase 4b experiment with logging.

---

## F. Clarity issues (nice to have)

### F1. "fingerprint encoding containment but not ordering"
**Location:** §4.4 line 326.
**Issue:** Phrasing is opaque.
**Fix:** "the graph's fingerprints encode *which* winners fire but not *in what temporal order*, making the signal poorly suited to temporal-ordering queries."

### F2. "Tautological gains where the graph rediscovers the encoder's own clusters"
**Location:** §1 line 51-52.
**Issue:** "Tautological" is the right word but unclear without example.
**Fix:** "tautological gains where the graph rediscovers the encoder's own clusters (a correlation artifact: if the graph is trained on activations-of-encoded-inputs, its structure will naturally align with encoder similarity and appear to add signal)".

### F3. Appendix B phase-to-commit mapping is dense
**Location:** Appendix B.
**Suggestion:** Convert to a table with columns {Phase, Short description, Commit, Section referenced}.

### F4. Abstract is a single long paragraph
**Location:** Abstract.
**Suggestion:** Break into 2-3 short paragraphs for readability.

---

## G. Out-of-scope items for the draft (flag for future)

These are real gaps but not appropriate to fix in the draft itself:

- **No figures** (E1): figures are a polish step before submission.
- **Related work citations** (D8): bibtex compilation before submission.
- **Ablation vs structure control** (E3): additional experiment, not a draft edit.
- **Broader encoder comparison** (§8 limitation already notes this).

---

## Prioritized edit list

If doing this in one session:

1. **A1, A2, A4** (factual errors in tables / numbers) — 10 min
2. **A3** (per-edge → per-associator) — 2 min
3. **A5** (drop "275%" or replace with verifiable number) — 5 min
4. **A6** (reconcile 14 → 16 experiment count) — 5 min
5. **B1, B2** (capacity-matched claim, adaptation window presentation) — 15 min
6. **B3** (update abstract to mention env findings) — 10 min
7. **B4** (introduce or remove Phase 8 reference) — 5 min
8. **C1** (soften "substrate specifically" claim) — 5 min
9. **C2** (soften or re-run "cross-session recall") — 5 min
10. **C3, C4, C5** (minor over-claim fixes) — 5 min
11. **D1, D2, D4, D5** (method gaps, appendix split) — 15 min
12. **D6, D7** (limitations + consolidation context) — 10 min
13. **E2, E4** (methodology framing + tanh note) — 10 min
14. **F1, F2, F3, F4** (clarity) — 10 min

**Total estimated:** ~2 hours for a polished revision, all verifiable claims.

**Deferred to pre-submission:**
- D8 (citations)
- E1 (figures)
- E3 (structure-control experiment)
- E5 (Phase 4b figure)
