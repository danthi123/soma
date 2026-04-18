# Developmental SOMA: Proof-of-Concept Architecture

> **Goal:** Build a "virtual baby" that develops intelligence through text
> interaction. SOMA is the brain (learns, adapts, grows). An LLM is the
> mouth (translates internal state to language). Development is visible:
> early interactions are shallow; mature interactions show memory,
> association, and anticipation.

**Date:** 2026-04-17

---

## Core Architecture

```
Human types text
       |
       v
  TextEncoder (BPE -> embeddings)
       |
       v
  SOMA Graph
    - activates nodes based on input
    - updates Hebbian weights
    - computes prediction error (vs predicted activation)
    - triggers neurogenesis if error stays high
    - prunes unused nodes
    - stores to episodic memory
       |
       v
  verbalize_state()
    - top-K activated nodes + labels
    - working memory contents
    - curiosity/novelty score
    - strongest recent associations
    - developmental stage
       |
       v
  Small LLM (1-3B, e.g. Qwen2-1.5B)
    - receives ONLY SOMA's context packet
    - never sees raw input directly
    - generates natural language response
       |
       v
  Human reads response
```

### Key Principle: SOMA is the Brain, LLM is the Mouth

The LLM does NOT think, remember, or learn. It translates. SOMA does all
learning, memory, and association. The LLM is deliberately weak (1-3B)
so it cannot compensate for missing SOMA state — responses are only as
good as SOMA's development allows.

---

## Design Question 1: How the LLM Reads SOMA's State

### Approach: Structured Context Packet

After each input, `verbalize_state()` produces a structured block:

```
[SOMA Internal State]
Developmental Stage: early-plasticity (step 847)
Novelty Score: 0.83 (HIGH — unfamiliar concept cluster)

Active Associations (top 5):
  1. "cooking" <-> "Italian food" (weight: 0.91, formed step 200)
  2. "travel" <-> "Italy" (weight: 0.87, formed step 340)
  3. "cooking" <-> "travel" (weight: 0.62, formed step 510)
  4. ...

Working Memory (3/32 slots):
  - "User mentioned wanting to visit Rome" (step 840, decay: 0.95)
  - "User enjoys making pasta" (step 812, decay: 0.78)
  - "User asked about Italian language" (step 845, decay: 0.98)

Episodic Recall (triggered by input):
  - [step 200] "User described learning to cook from grandmother"
  - [step 340] "User talked about dream trip to Europe"

Curiosity Signals:
  - "Italian language" — HIGH novelty (no existing node cluster)
  - "cooking" — LOW novelty (well-established cluster)
```

The LLM receives this block + a system prompt:

```
You are the voice of a developing mind. Your responses must reflect
ONLY the internal state provided above. If the state shows high
novelty, express curiosity. If associations are strong, make
connections. If working memory is sparse, be brief. You are not
a knowledgeable assistant — you are a developing intelligence
expressing what it currently understands.
```

### Why This Works

- Zero fine-tuning — works with any LLM API
- Human-readable state (debuggable)
- SOMA's development is directly visible in the state block
- LLM behavior changes as state changes (sparse state -> sparse responses)

### Future Evolution

Once the PoC validates the concept, deeper integration options:
- Embedding projection (soft-token prefixes)
- Cross-attention over node embeddings
- Fine-tuned small LLM on SOMA state -> response pairs

---

## Design Question 2: What Drives SOMA to Learn

### Approach: Next-Input Activation Prediction

**Core loop for each input:**

1. SOMA predicts which nodes will activate for the next input
   (based on current state + conversation context)
2. Next input arrives, actually activates nodes
3. Prediction error = difference between predicted and actual activation
4. Error signal drives all learning mechanisms:

```
Prediction Error
       |
       +---> Hebbian update: strengthen edges that predicted correctly
       |
       +---> Neurogenesis: if error is persistently high for this
       |     input type, create new nodes (graph lacks structure)
       |
       +---> Pruning: remove nodes that never contribute to
       |     reducing prediction error
       |
       +---> Curiosity: prediction error IS the novelty signal
       |     (high error = novel input = explore more)
       |
       +---> Consolidation: during "sleep," replay high-error
              episodes to strengthen predictive pathways
```

### Why Prediction, Not Reconstruction or Pure Hebbian

- **Pure Hebbian** (option 3): Creates clusters but no hierarchy, no
  abstraction. The graph organizes statistically but without pressure
  toward useful representations.

- **Reconstruction** (option 2): Creates autoencoders. The graph learns
  to compress/decompress but never develops anticipatory structure.
  No "what comes next" capability.

- **Prediction** (option 1): Creates anticipatory structure. The graph
  learns temporal patterns, develops expectations, and is surprised by
  genuinely novel input. This is what biological brains do (Free Energy
  Principle / predictive processing).

### Implementation Detail

```python
# Pseudocode for the prediction loop
class PredictiveSOMA:
    def process_input(self, text: str) -> PredictionResult:
        # 1. Encode input
        embedding = self.encoder(text)

        # 2. Compare with prediction from last step
        actual_activation = self.graph.execute(embedding)
        prediction_error = self.compute_error(
            self.last_prediction, actual_activation
        )

        # 3. Learn from error
        self.hebbian_update(prediction_error)
        if self.should_grow(prediction_error):
            self.neurogenesis(embedding)
        if self.should_prune():
            self.prune_unused()

        # 4. Predict next input's activation
        self.last_prediction = self.graph.predict_next(actual_activation)

        # 5. Store experience
        self.episodic_memory.store(text, actual_activation, prediction_error)

        return PredictionResult(
            activation=actual_activation,
            prediction_error=prediction_error,
            novelty=prediction_error.mean(),
        )
```

### Developmental Staging (Critical Periods)

- **Early (steps 0-1000):** High prediction error tolerance, aggressive
  neurogenesis, low pruning. Graph grows rapidly and messily. Like a
  baby's brain overproducing synapses.

- **Middle (steps 1000-5000):** Moderate error tolerance. Pruning
  increases. Consolidation cycles become important. Graph organizes
  into clusters. Myelination strengthens reliable pathways.

- **Mature (steps 5000+):** Low error tolerance (only truly novel
  input triggers growth). Heavy pruning. Strong myelinated pathways.
  Consolidation dominant. Like an adult brain — stable but still
  capable of learning.

This maps directly to SOMA's existing `DevelopmentSchedule` and
critical periods infrastructure.

---

## Design Question 3: Preventing LLM Dominance

### Approach: Information Bottleneck + Weak LLM + Ablation

**Three reinforcing mechanisms:**

#### A. Weak LLM (Architectural)

Use a 1-3B parameter model (Qwen2-1.5B, Phi-3-mini, or similar).
These models cannot compensate for missing context — they genuinely
need SOMA's associations and memory to produce coherent responses
about conversation history and learned patterns.

Available via Ollama locally. No API costs.

#### B. Information Bottleneck (Architectural)

The LLM NEVER sees the raw user input. It receives:
1. SOMA's verbalized state (associations, memory, curiosity)
2. A generation instruction ("respond as a developing mind")

The raw input is processed only by SOMA. The LLM works with SOMA's
interpretation of the input, not the input itself.

```
User: "I made carbonara last night, reminded me of my trip to Rome"
        |
        v
     [SOMA processes — activates cooking, Italy, travel nodes;
      forms new edge: carbonara <-> Rome; high activation on
      grandmother-cooking episodic memory]
        |
        v
     [LLM receives SOMA state, NOT the raw text]
        |
        v
LLM: "The connection between cooking and that place is getting
      stronger. There's something familiar about making food and
      the person who taught you..."
```

The LLM's response reflects SOMA's associations (cooking <->
grandmother) without having seen the user mention either.

#### C. Ablation Delta (Evaluation)

Every evaluation runs two conditions:
1. **SOMA-attached:** LLM receives full SOMA state
2. **SOMA-zeroed:** LLM receives empty/default state

The difference is the "development delta" — the measurable
contribution of SOMA's learning. This is the primary metric.

```python
def evaluate_development(soma, llm, test_inputs):
    # With SOMA
    soma_responses = []
    for text in test_inputs:
        state = soma.process_and_verbalize(text)
        response = llm.generate(state)
        soma_responses.append(response)

    # Without SOMA (zeroed state)
    zero_responses = []
    for text in test_inputs:
        state = soma.empty_state()
        response = llm.generate(state)
        zero_responses.append(response)

    # Delta = SOMA's contribution
    return compute_delta(soma_responses, zero_responses)
```

#### D. Task Design (Evaluation)

Test on content the LLM cannot know from training:
- Facts shared during the conversation
- Associations formed across sessions
- Patterns that emerged over time
- Novel terminology invented in-session

---

## What "Development" Looks Like — Observable Milestones

### Stage 1: Blank Slate (steps 0-100)
- Graph: ~30 nodes (initial seed), sparse edges
- Behavior: Generic, confused responses. No memory of prior turns.
- Curiosity: Everything is novel (high prediction error everywhere)
- LLM output: Short, vague, disoriented

### Stage 2: Pattern Recognition (steps 100-500)
- Graph: ~100-200 nodes, emerging clusters
- Behavior: Starts recognizing recurring topics. "You've mentioned
  cooking before" type connections.
- Curiosity: Familiar topics have low novelty, new topics trigger
  growth
- LLM output: Shows awareness of conversation history

### Stage 3: Association (steps 500-2000)
- Graph: ~300-500 nodes, dense clusters with cross-cluster edges
- Behavior: Makes connections across topics and sessions.
  "Cooking and your grandmother" type associations.
- Curiosity: Surprised by genuinely new concepts, not by variations
  on known themes
- Consolidation: "Sleep" cycles visibly reorganize the graph
- LLM output: Richer, more connected responses

### Stage 4: Anticipation (steps 2000+)
- Graph: ~500-1000 nodes, heavily pruned and myelinated
- Behavior: Predicts conversation direction. "I expected you'd
  bring up travel after mentioning Italian food."
- Curiosity: Highly selective — only truly novel input triggers
  growth
- Consolidation: Major reorganization events become rare
- LLM output: Demonstrates genuine understanding of the user's
  patterns and preferences

---

## Implementation Plan — Phase 1 (Proof of Concept)

### What We Build

1. **PredictiveSOMA wrapper** — wraps existing SOMA system with
   prediction loop (predict next activation, compute error, learn)

2. **verbalize_state()** — function that serializes graph state
   to structured text for LLM consumption

3. **InteractionLoop** — main loop: input -> SOMA -> verbalize ->
   LLM -> output. Handles consolidation scheduling.

4. **DevelopmentTracker** — logs graph metrics over time (node
   count, edge count, clustering coefficient, prediction error,
   novelty distribution)

5. **Ablation harness** — runs with/without SOMA state to measure
   development delta

### What We Reuse (from existing SOMA)

- `src/soma/core/` — Graph engine, nodes, edges, execution
- `src/soma/growth/` — Neurogenesis, synaptogenesis, pruning,
  myelination
- `src/soma/metacognition/` — Curiosity, homeostatic regulation,
  DevelopmentSchedule
- `src/soma/consolidation/` — Consolidation cycles
- `src/soma/memory/` — Working memory, episodic memory
- `src/soma/io/text_encoder.py` — TextEncoder, BPE tokenizer
- `src/soma/core/config.py` — SOMAConfig

### What We Build New

- `src/soma/developmental/` — New module
  - `prediction.py` — PredictionLoop (predict, error, learn)
  - `verbalize.py` — verbalize_state() function
  - `interaction.py` — InteractionLoop (human <-> SOMA <-> LLM)
  - `tracker.py` — DevelopmentTracker (metrics over time)
  - `ablation.py` — Ablation harness

### What We DON'T Need (from existing SOMA)

- `src/soma/memory/api.py` — MemoryLayer API (agent memory pivot)
- `src/soma/deploy/` — Deployment infrastructure
- `src/soma/training/` — Verbalizer bootstrap

### Tech Stack

- SOMA core (PyTorch, existing code)
- Ollama (local LLM hosting, already set up)
- Qwen2-1.5B or similar small model (via Ollama)
- sentence-transformers (embedding, already in use)
- CLI interaction initially (web UI later)

---

## Success Criteria

The PoC succeeds if:

1. **Development is measurable:** Graph metrics change meaningfully
   over interaction (not random walk)

2. **Development affects output:** Ablation delta is significant —
   SOMA-attached responses are measurably better than SOMA-zeroed
   on conversation-specific tasks

3. **Brain mechanisms contribute:** Neurogenesis, pruning, and
   consolidation each independently improve development (ablation
   per mechanism)

4. **A static system can't match it:** A frozen graph (no learning)
   performs worse than a developing one over extended interaction

---

## Open Questions

1. **Graph-to-prediction bridge:** How exactly does the graph
   "predict" next activation? Options: linear projection from
   current state, learned transition model, or simple momentum
   (predict same activation continues). Start simple.

2. **Consolidation trigger:** When does the system "sleep"?
   After N interactions? When prediction error plateaus?
   On explicit command? Start with periodic (every N steps).

3. **Node labeling:** For verbalize_state() to produce meaningful
   text, nodes need semantic labels. How are these assigned?
   Options: cluster centroid text, most-activating input, or
   learned label. Start with most-activating input text.

4. **Scale:** How many interactions before development is visible?
   Biological analogy suggests hundreds to thousands, not millions.
   But we need to test empirically.

5. **Evaluation beyond ablation:** What does a "developmental
   benchmark" look like? Track association quality, prediction
   accuracy, adaptation speed after topic change, memory retention
   after consolidation.
