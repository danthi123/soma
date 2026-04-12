# SOMA: Self-Organizing Memory Architecture
## A Technical Whitepaper for a Brain-Inspired Developmental AI System

**Version:** 1.0
**Target Platform:** Single NVIDIA RTX 3090 (24GB VRAM)
**Implementation Language:** Python (PyTorch)

---

## 1. Executive Summary

SOMA is a novel AI architecture designed from first principles around the brain's developmental mechanisms rather than the transformer paradigm. Unlike conventional models that are trained on massive static datasets and then deployed frozen, SOMA is a dynamic system that grows through interaction. It starts from a minimal seed architecture and self-organizes over time — forming new connections, pruning unused ones, consolidating memories, and developing increasingly complex capabilities through staged exposure to experience.

SOMA is not a language model. It is a general-purpose adaptive processing system with multimodal input/output that can develop language capabilities (among other skills) when exposed to linguistic interaction. The architecture is defined by five core brain-inspired principles:

1. Structural plasticity — The network topology changes over time (growth, pruning, myelination)
2. Complementary memory systems — Separate fast episodic and slow parametric memory
3. Consolidation through replay — Offline integration of experiences into long-term knowledge
4. Developmental curriculum — Staged exposure that builds capabilities incrementally
5. Intrinsic motivation — Self-generated learning signals that drive active exploration

The system is designed to be feasible on a single RTX 3090. It starts at ~1K nodes and grows organically, never exceeding the available VRAM.

---

## 2. Architecture Overview

### 2.1 Design Philosophy

Traditional neural networks are fixed computational graphs trained via gradient descent on static datasets. SOMA inverts this paradigm:

- The graph structure is dynamic — nodes and edges are created and destroyed during the model's lifetime
- Learning never stops — there is no train/deploy split; the system is always adapting
- Memory is explicitly structured — three distinct memory systems serve different functional roles
- Development is staged — capabilities emerge in a biologically-inspired progression
- The system is self-regulating — homeostatic mechanisms maintain stability without external tuning

### 2.2 High-Level Component Map

```
+-------------------------------------------------------------+
|                        SOMA System                          |
|                                                             |
|  +----------+   +------------------+   +--------------+    |
|  | I/O      |   | Processing Core  |   | Memory       |    |
|  | Boundary |<->| (Dynamic Graph)  |<->| Systems      |    |
|  |          |   |                  |   |              |    |
|  | - Text   |   | - Sensor Nodes   |   | - Working    |    |
|  | - Image  |   | - Associator     |   | - Episodic   |    |
|  | - Audio  |   |   Nodes          |   | - Parametric |    |
|  | - Action |   | - Integrator     |   |              |    |
|  |          |   |   Nodes          |   |              |    |
|  |          |   | - Output Nodes   |   |              |    |
|  +----------+   +------------------+   +--------------+    |
|                                                             |
|  +------------------+   +------------------------------+   |
|  | Growth Engine     |   | Meta-Cognitive Module        |   |
|  |                   |   |                              |   |
|  | - Synaptogenesis  |   | - Curiosity / Surprise       |   |
|  | - Pruning         |   | - Homeostatic Regulation     |   |
|  | - Myelination     |   | - Consolidation Scheduler    |   |
|  | - Critical Periods|   | - Development Stage Tracker  |   |
|  +------------------+   +------------------------------+   |
+-------------------------------------------------------------+
```

---

## 3. Processing Core: The Dynamic Graph

### 3.1 Node Specification

The processing core is a directed graph G = (V, E) where nodes and edges are created and destroyed dynamically. Every node is a small, self-contained processing unit.

#### Node Data Structure

NOTE: In implementation, Node should be an nn.Module (not a pure dataclass) since it holds
learnable parameters (W1, b1, W2, b2). The dataclass notation below is for specification clarity.
The `weights` and `bias` fields below should be implemented as separate W1, b1, W2, b2 nn.Parameter
tensors matching the forward() method. All methods that reference `current_step` should receive it
as an explicit parameter.

```python
@dataclass  # Implement as nn.Module
class Node:
    id: str                          # Unique identifier (UUID)
    node_type: NodeType              # SENSOR | ASSOCIATOR | INTEGRATOR | OUTPUT
    # W1: nn.Parameter (input_dim x hidden_dim), b1: nn.Parameter (hidden_dim,)
    # W2: nn.Parameter (hidden_dim x output_dim), b2: nn.Parameter (output_dim,)
    activation_history: RingBuffer   # Recent activation values (fixed-size circular buffer)
    activation_history: RingBuffer   # Recent activation values (fixed-size circular buffer)
    activation_ema: float            # Exponential moving average of activation magnitude
    creation_step: int               # When this node was created
    last_active_step: int            # Last step where activation exceeded threshold
    maturity: float                  # 0.0 (newborn) to 1.0 (fully mature) -- affects learning rate
    position: torch.Tensor           # Soft "position" in latent space (used for locality-biased wiring)

    # Internal MLP specification
    input_dim: int                   # Number of input connections this node accepts
    hidden_dim: int                  # Internal hidden dimension (typically 2x input_dim)
    output_dim: int                  # Dimension of output activation vector

    # Homeostatic parameters
    target_activation: float         # Target mean activation level (set at creation)
    gain: float                      # Multiplicative gain factor (adjusted by homeostasis)
```

#### Node Types

| Type | Role | Biological Analog | Count at Init |
|------|------|-------------------|---------------|
| SENSOR | Receives external input (one per modality/channel) | Primary sensory cortex | Fixed (set by I/O config) |
| ASSOCIATOR | Learns associations between patterns | Association cortex | Starts small, grows |
| INTEGRATOR | Combines information across regions of the graph | Prefrontal / parietal cortex | Starts very small, grows later |
| OUTPUT | Produces external outputs (one per output modality) | Motor cortex | Fixed (set by I/O config) |

#### Node Internal Computation

Each node implements a small MLP with residual connection:

```python
def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    inputs: dict mapping source_node_id -> (activation_vector * edge_weight)
    Returns: output activation vector
    """
    # Aggregate inputs (sum of weighted incoming activations)
    x = torch.zeros(self.input_dim)
    for source_id, weighted_activation in inputs.items():
        x += weighted_activation

    # Small MLP with residual
    h = F.gelu(self.W1 @ x + self.b1)          # input_dim -> hidden_dim
    h = self.W2 @ h + self.b2                    # hidden_dim -> output_dim

    # Apply homeostatic gain
    h = h * self.gain

    # Residual connection (if dims match)
    if self.input_dim == self.output_dim:
        h = h + x[:self.output_dim]

    # Record activation
    activation_magnitude = h.norm().item()
    self.activation_history.append(activation_magnitude)
    self.activation_ema = 0.99 * self.activation_ema + 0.01 * activation_magnitude
    self.last_active_step = current_step

    return h
```

Default dimensions at initialization:
- SENSOR: input_dim = modality-specific (e.g., 64 for text token embeddings), hidden_dim = 128, output_dim = 64
- ASSOCIATOR: input_dim = 64, hidden_dim = 128, output_dim = 64
- INTEGRATOR: input_dim = 128, hidden_dim = 256, output_dim = 128
- OUTPUT: input_dim = 64, hidden_dim = 128, output_dim = modality-specific

### 3.2 Edge Specification

Edges carry weighted activation vectors between nodes.

```python
@dataclass
class Edge:
    id: str                          # Unique identifier
    source_id: str                   # Source node
    target_id: str                   # Target node
    weight: float                    # Scalar weight (learnable, initialized ~0.1)
    projection: torch.Tensor         # Linear projection matrix (source.output_dim -> target.input_dim)
                                     # Only needed if source and target have different dims
    coactivation_count: int          # Number of times source and target were both active
    last_active_step: int            # Last step where this edge carried nonzero signal
    creation_step: int               # When this edge was created
    strength: float                  # Running measure of utility (EMA of |signal * gradient|)
```

#### Edge Forward Pass

```python
def transmit(self, source_activation: torch.Tensor) -> torch.Tensor:
    """Transform and weight the source activation for delivery to target."""
    if self.projection is not None:
        signal = self.projection @ source_activation
    else:
        signal = source_activation
    return signal * self.weight
```

### 3.3 Graph Execution

The graph is executed in waves, not layers. SOMA processes in propagation waves from input to output, following the topological structure of whatever graph currently exists.

```python
def execute_graph(graph: Graph, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Execute one full forward pass through the dynamic graph.

    1. Inject inputs into SENSOR nodes
    2. Propagate activations wave-by-wave (topological order)
    3. Collect outputs from OUTPUT nodes
    """
    # Step 1: Inject inputs
    for modality, data in inputs.items():
        sensor_node = graph.get_sensor(modality)
        sensor_node.set_input(data)

    # Step 2: Compute execution order via topological sort
    execution_order = graph.topological_sort()

    # Step 3: Propagate
    activations = {}  # node_id -> activation_vector

    for node_id in execution_order:
        node = graph.nodes[node_id]

        if node.node_type == NodeType.SENSOR:
            activations[node_id] = node.get_input_activation()
            continue

        # Gather inputs from all incoming edges
        incoming = {}
        for edge in graph.get_incoming_edges(node_id):
            if edge.source_id in activations:
                source_act = activations[edge.source_id]
                incoming[edge.source_id] = edge.transmit(source_act)

        # Skip if no active inputs (node stays dormant)
        if not incoming:
            continue

        # Forward pass through node
        activations[node_id] = node.forward(incoming)

    # Step 4: Collect outputs
    outputs = {}
    for modality, output_node in graph.output_nodes.items():
        if output_node.id in activations:
            outputs[modality] = activations[output_node.id]

    return outputs
```

#### Handling Cycles

The graph may develop cycles (feedback connections) through growth. These are handled by:
1. Using activations from the previous timestep for any back-edges (edges that point backward in the topological order)
2. Maintaining a previous_activations buffer that stores the last timestep's values
3. Back-edges are flagged during topological sort and treated as recurrent connections

This gives the system implicit recurrence without requiring explicit loop mechanisms.

### 3.4 Learning Rule: Augmented Backpropagation

SOMA uses standard backpropagation through the active subgraph, but augmented with Hebbian reinforcement:

```python
def update_step(graph, loss, global_step, activations, config):
    """
    Combined gradient-based and Hebbian update.
    activations: Dict[str, torch.Tensor] — node_id -> activation from the forward pass.
    config: SOMAConfig — provides BASE_LR, YOUTH_LR_MULTIPLIER, etc.
    """
    # 1. Standard backpropagation through the active subgraph
    loss.backward()

    for node in graph.active_nodes():
        # Scale learning rate by inverse maturity (young nodes learn faster)
        lr = BASE_LR * (1.0 + (1.0 - node.maturity) * YOUTH_LR_MULTIPLIER)

        # Standard gradient update
        with torch.no_grad():
            for param in node.parameters():
                if param.grad is not None:
                    param -= lr * param.grad

    # 2. Hebbian edge weight update (fire-together-wire-together)
    for edge in graph.active_edges():
        source_act = activations[edge.source_id].norm()
        target_act = activations[edge.target_id].norm()

        # Coactivation strengthening
        if source_act > ACTIVATION_THRESHOLD and target_act > ACTIVATION_THRESHOLD:
            edge.coactivation_count += 1
            edge.weight += HEBBIAN_LR * source_act * target_act

        # Clip edge weight to prevent explosion
        edge.weight = torch.clamp(edge.weight, -MAX_EDGE_WEIGHT, MAX_EDGE_WEIGHT)

        # Update edge strength (utility tracking for pruning)
        edge.strength = 0.999 * edge.strength + 0.001 * abs(edge.weight * source_act)

    # 3. Advance node maturity
    for node in graph.active_nodes():
        node.maturity = min(1.0, node.maturity + MATURITY_INCREMENT)

    # 4. Homeostatic gain adjustment
    for node in graph.all_nodes():
        if node.activation_ema > node.target_activation * 1.2:
            node.gain *= 0.999  # Too active, reduce gain
        elif node.activation_ema < node.target_activation * 0.8:
            node.gain *= 1.001  # Too quiet, increase gain
        node.gain = max(0.1, min(10.0, node.gain))  # Clamp
```

---

## 4. Memory Systems

### 4.1 Working Memory

Working memory is a small, fixed-capacity buffer that supports active processing. Analogous to the prefrontal cortex's ability to hold and manipulate a few items simultaneously.

```python
class WorkingMemory:
    """
    Fixed-capacity memory with attention-based read and gated write.
    Capacity: 16-64 slots (configurable).
    Each slot holds a vector of dimension wm_dim.
    Content decays unless actively refreshed.
    """

    def __init__(self, num_slots: int = 32, wm_dim: int = 128, decay_rate: float = 0.95):
        self.slots = torch.zeros(num_slots, wm_dim)       # Memory content
        self.usage = torch.zeros(num_slots)                 # How "full" each slot is (0-1)
        self.age = torch.zeros(num_slots, dtype=torch.long) # Steps since last write
        self.decay_rate = decay_rate

        # Learnable parameters
        self.query_proj = nn.Linear(wm_dim, wm_dim)         # For read attention
        self.write_gate = nn.Linear(wm_dim * 2, 1)          # Decides whether to write
        self.erase_gate = nn.Linear(wm_dim, num_slots)       # Decides what to erase

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Attention-based read. Returns weighted combination of slots."""
        q = self.query_proj(query)
        attention = F.softmax(self.slots @ q / math.sqrt(self.slots.shape[-1]), dim=0)
        return (attention.unsqueeze(-1) * self.slots).sum(dim=0)

    def write(self, content: torch.Tensor, context: torch.Tensor):
        """Gated write to least-used slot."""
        gate_input = torch.cat([content, context])
        write_prob = torch.sigmoid(self.write_gate(gate_input))

        if write_prob > 0.5:
            slot_idx = torch.argmin(self.usage)
            self.slots[slot_idx] = content
            self.usage[slot_idx] = 1.0
            self.age[slot_idx] = 0

    def step(self):
        """Called each timestep. Applies decay."""
        self.usage *= self.decay_rate
        self.age += 1
        fade_mask = (self.usage < 0.1).float().unsqueeze(-1)
        self.slots *= (1.0 - fade_mask * 0.1)
```

### 4.2 Episodic Memory

Episodic memory performs fast, one-shot encoding of experiences. Analogous to the hippocampus.

```python
class EpisodicMemory:
    """
    Content-addressable memory with fast write, similarity-based read, and decay.
    Capacity: configurable (default 10,000 episodes).
    """

    def __init__(self, capacity: int = 10000, key_dim: int = 128, value_dim: int = 256):
        self.capacity = capacity
        self.keys = torch.zeros(capacity, key_dim)
        self.values = torch.zeros(capacity, value_dim)
        self.surprise = torch.zeros(capacity)
        self.timestamps = torch.zeros(capacity, dtype=torch.long)
        self.valid = torch.zeros(capacity, dtype=torch.bool)
        self.write_head = 0

        self.key_encoder = nn.Sequential(
            nn.Linear(value_dim, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
        )

    def encode(self, experience: torch.Tensor, prediction_error: float):
        """Store a new experience."""
        key = self.key_encoder(experience.detach())
        self.keys[self.write_head] = key
        self.values[self.write_head] = experience.detach()
        self.surprise[self.write_head] = prediction_error
        self.timestamps[self.write_head] = current_step
        self.valid[self.write_head] = True
        self.write_head = (self.write_head + 1) % self.capacity

    def retrieve(self, query: torch.Tensor, top_k: int = 5) -> List[Tuple[torch.Tensor, float]]:
        """Retrieve most similar episodes."""
        if not self.valid.any():
            return []
        valid_keys = self.keys[self.valid]
        valid_values = self.values[self.valid]
        similarities = F.cosine_similarity(query.unsqueeze(0), valid_keys, dim=-1)
        top_indices = similarities.topk(min(top_k, len(valid_keys))).indices
        results = []
        for idx in top_indices:
            results.append((valid_values[idx], similarities[idx].item()))
        return results

    def sample_for_replay(self, batch_size: int = 32) -> List[torch.Tensor]:
        """
        Sample experiences for consolidation replay.
        Prioritizes: high surprise, moderate recency (not too old, not too new).
        """
        if not self.valid.any():
            return []
        valid_mask = self.valid.float()
        ages = (current_step - self.timestamps).float()
        age_weight = torch.exp(-((ages - 1000) / 2000) ** 2)
        priority = self.surprise * age_weight * valid_mask
        priority = priority / (priority.sum() + 1e-8)
        indices = torch.multinomial(priority, min(batch_size, int(valid_mask.sum())), replacement=False)
        return [self.values[i] for i in indices]
```

### 4.3 Parametric Memory (The Graph Weights)

Parametric memory is simply the weights of all nodes and edges in the graph. No separate data structure -- the graph IS the parametric memory. Updated through:
1. Online learning -- small gradient updates during interaction (Section 3.4)
2. Consolidation replay -- larger updates during sleep/replay phases (Section 6)

---

## 5. Growth Engine

### 5.1 Synaptogenesis (New Connection Formation)

New edges are created based on co-activation patterns (Hebbian: fire together, wire together).

```python
def synaptogenesis(graph: Graph, activations: Dict[str, torch.Tensor], step: int):
    """
    Check for potential new connections based on co-activation.
    Called periodically (every N steps, not every step -- too expensive).
    """
    active_nodes = [
        nid for nid, act in activations.items()
        if act.norm() > ACTIVATION_THRESHOLD
    ]

    for i, source_id in enumerate(active_nodes):
        for target_id in active_nodes[i+1:]:
            if graph.has_edge(source_id, target_id):
                continue

            source_act = activations[source_id].norm()
            target_act = activations[target_id].norm()
            coact_strength = source_act * target_act

            distance = (graph.nodes[source_id].position - graph.nodes[target_id].position).norm()
            locality_bonus = torch.exp(-distance / LOCALITY_SCALE)

            connection_prob = coact_strength * locality_bonus * SYNAPTOGENESIS_RATE

            if torch.rand(1).item() < connection_prob:
                graph.add_edge(Edge(
                    id=generate_uuid(),
                    source_id=source_id,
                    target_id=target_id,
                    weight=0.01 * torch.randn(1).item(),
                    projection=create_projection_if_needed(
                        graph.nodes[source_id].output_dim,
                        graph.nodes[target_id].input_dim
                    ),
                    coactivation_count=1,
                    last_active_step=step,
                    creation_step=step,
                    strength=0.01,
                ))
```

### 5.2 Neurogenesis (New Node Creation)

New nodes are created when the system encounters persistent prediction errors that existing nodes cannot resolve.

```python
def neurogenesis(graph: Graph, recent_errors: List[float], step: int):
    """Create new ASSOCIATOR nodes when prediction errors are persistently high."""
    mean_recent_error = sum(recent_errors[-100:]) / max(len(recent_errors[-100:]), 1)
    baseline_error = sum(recent_errors[-1000:]) / max(len(recent_errors[-1000:]), 1)

    error_trend = mean_recent_error / (baseline_error + 1e-8)

    if error_trend > NEUROGENESIS_THRESHOLD and len(graph.nodes) < MAX_NODES:
        active_positions = [
            graph.nodes[nid].position
            for nid in graph.most_active_nodes(k=10)
        ]

        if active_positions:
            new_position = torch.stack(active_positions).mean(dim=0)
            new_position += torch.randn_like(new_position) * POSITION_JITTER
        else:
            new_position = torch.randn(POSITION_DIM) * 0.1

        new_node = Node(
            id=generate_uuid(),
            node_type=NodeType.ASSOCIATOR,
            # ... (see full spec in Section 3.1)
            maturity=0.0,  # Newborn -- high learning rate
            position=new_position,
        )

        graph.add_node(new_node)

        nearby_nodes = graph.get_nearest_nodes(new_position, k=5)
        for neighbor_id in nearby_nodes:
            graph.add_edge(create_edge(neighbor_id, new_node.id, step))
            graph.add_edge(create_edge(new_node.id, neighbor_id, step))

        return new_node
    return None
```

### 5.3 Pruning

Connections and nodes that are not useful get removed.

```python
def pruning(graph: Graph, step: int):
    """Remove weak edges and orphaned nodes. Called periodically."""
    edges_to_remove = []

    for edge in graph.all_edges():
        age = step - edge.creation_step
        if age < PRUNING_GRACE_PERIOD:
            continue
        time_since_active = step - edge.last_active_step
        if edge.strength < EDGE_STRENGTH_THRESHOLD and time_since_active > INACTIVITY_THRESHOLD:
            edges_to_remove.append(edge.id)

    for edge_id in edges_to_remove:
        graph.remove_edge(edge_id)

    # Remove orphaned ASSOCIATOR/INTEGRATOR nodes (no remaining edges)
    nodes_to_remove = []
    for node in graph.all_nodes():
        if node.node_type in (NodeType.SENSOR, NodeType.OUTPUT):
            continue  # Never prune I/O nodes
        incoming = graph.get_incoming_edges(node.id)
        outgoing = graph.get_outgoing_edges(node.id)
        if len(incoming) == 0 and len(outgoing) == 0:
            nodes_to_remove.append(node.id)

    for node_id in nodes_to_remove:
        graph.remove_node(node_id)

    return len(edges_to_remove), len(nodes_to_remove)
```

### 5.4 Myelination (Circuit Consolidation)

When a group of nodes consistently activates in sequence, they can be merged into a single, more efficient node.

```python
def myelination(graph: Graph, step: int):
    """Detect frequently-used linear chains and consolidate them."""
    chains = detect_linear_chains(graph, min_length=2, max_length=4)

    for chain in chains:
        chain_edges = [graph.get_edge(chain[i], chain[i+1]) for i in range(len(chain)-1)]
        all_mature = all(e.strength > MYELINATION_STRENGTH_THRESHOLD for e in chain_edges)
        all_old = all((step - e.creation_step) > MYELINATION_AGE_THRESHOLD for e in chain_edges)

        if all_mature and all_old:
            merged_node = compose_nodes(
                [graph.nodes[nid] for nid in chain],
                chain_edges, step
            )
            redirect_external_edges(graph, chain, merged_node)
            for nid in chain:
                graph.remove_node(nid)
            graph.add_node(merged_node)
```

### 5.5 Critical Periods

```python
@dataclass
class CriticalPeriod:
    name: str
    start_step: int
    peak_step: int
    end_step: int
    affected_node_types: List[NodeType]
    plasticity_multiplier: float
    synaptogenesis_multiplier: float

class DevelopmentSchedule:
    periods: List[CriticalPeriod] = [
        CriticalPeriod("sensory_discrimination", 0, 5000, 20000,
                        [NodeType.SENSOR], 3.0, 5.0),
        CriticalPeriod("association_formation", 5000, 25000, 50000,
                        [NodeType.ASSOCIATOR], 2.5, 3.0),
        CriticalPeriod("integration", 20000, 50000, 100000,
                        [NodeType.INTEGRATOR], 2.0, 2.0),
        CriticalPeriod("output_refinement", 30000, 60000, 120000,
                        [NodeType.OUTPUT], 2.0, 1.5),
    ]

    def get_plasticity_multiplier(self, step: int, node_type: NodeType) -> float:
        multiplier = 1.0
        for period in self.periods:
            if node_type in period.affected_node_types:
                if period.start_step <= step <= period.end_step:
                    if step <= period.peak_step:
                        progress = (step - period.start_step) / (period.peak_step - period.start_step)
                    else:
                        progress = 1.0 - (step - period.peak_step) / (period.end_step - period.peak_step)
                    multiplier = max(multiplier, 1.0 + (period.plasticity_multiplier - 1.0) * progress)
        return multiplier
```

---

## 6. Consolidation System (Artificial Sleep)

### 6.1 Overview

Every CONSOLIDATION_INTERVAL steps (default: 1000), SOMA enters a consolidation phase. During this phase, it does not process new inputs. Instead:

1. Sample experiences from episodic memory (prioritized by surprise and moderate age)
2. Replay them through the graph with a reduced learning rate
3. Detect structural regularities (recurring patterns across episodes)
4. Prune weak connections
5. Run myelination checks

### 6.2 Consolidation Algorithm

```python
def consolidation_cycle(soma, num_replay_steps: int = 100):
    """Run one consolidation cycle (artificial sleep)."""
    replay_batch = soma.episodic_memory.sample_for_replay(batch_size=num_replay_steps)
    if not replay_batch:
        return

    consolidation_lr = BASE_LR * CONSOLIDATION_LR_RATIO

    replay_errors = []
    for experience in replay_batch:
        replay_input = soma.experience_to_input(experience)
        output = soma.forward(replay_input)
        replay_target = soma.experience_to_target(experience)
        loss = F.mse_loss(output, replay_target)

        loss.backward()
        for node in soma.graph.active_nodes():
            lr = consolidation_lr * (0.5 + 0.5 * node.maturity)
            with torch.no_grad():
                for param in node.parameters():
                    if param.grad is not None:
                        param -= lr * param.grad
                        param.grad.zero_()

        replay_errors.append(loss.item())

    # Structural maintenance
    pruning(soma.graph, soma.global_step)
    myelination(soma.graph, soma.global_step)

    # Optionally trigger neurogenesis if replay errors are high
    if sum(replay_errors) / len(replay_errors) > CONSOLIDATION_ERROR_THRESHOLD:
        neurogenesis(soma.graph, replay_errors, soma.global_step)

    soma.episodic_memory.decay_old_entries()
```

---

## 7. Meta-Cognitive Module

### 7.1 Curiosity / Intrinsic Motivation

```python
class CuriosityModule:
    """
    Tracks prediction error across different input domains/modalities
    and computes a curiosity score based on the RATE of improvement.
    """

    def __init__(self, num_domains: int = 8, window_size: int = 100):
        self.error_histories = {i: RingBuffer(window_size) for i in range(num_domains)}
        self.domain_classifier = nn.Linear(128, num_domains)

    def compute_curiosity(self, input_repr: torch.Tensor, prediction_error: float) -> float:
        """
        Returns a curiosity score for the current input.
        High curiosity = the model is learning fast in this domain.
        """
        domain_logits = self.domain_classifier(input_repr.detach())
        domain = domain_logits.argmax().item()
        self.error_histories[domain].append(prediction_error)

        history = self.error_histories[domain].get_all()
        if len(history) < 10:
            return 1.0

        recent_mean = sum(history[-10:]) / 10
        older_mean = sum(history[-50:-10]) / max(len(history[-50:-10]), 1)

        if older_mean > 0:
            learning_progress = (older_mean - recent_mean) / older_mean
        else:
            learning_progress = 0.0

        curiosity = max(0.0, learning_progress) * (recent_mean / (recent_mean + 0.1))
        return curiosity
```

### 7.2 Homeostatic Regulation

```python
class HomeostaticRegulator:
    def __init__(self, max_nodes: int = 50000, max_edges_per_node: float = 20.0):
        self.max_nodes = max_nodes
        self.max_edges_per_node = max_edges_per_node
        self.loss_ema = 0.0
        self.loss_variance_ema = 0.0
        self.global_lr_multiplier = 1.0

    def update(self, graph, current_loss: float):
        self.loss_ema = 0.99 * self.loss_ema + 0.01 * current_loss
        deviation = (current_loss - self.loss_ema) ** 2
        self.loss_variance_ema = 0.99 * self.loss_variance_ema + 0.01 * deviation

        std = math.sqrt(self.loss_variance_ema + 1e-8)
        if current_loss > self.loss_ema + 3 * std:
            self.global_lr_multiplier *= 0.5
        else:
            self.global_lr_multiplier = min(1.0, self.global_lr_multiplier * 1.01)

        num_nodes = len(graph.nodes)
        num_edges = len(graph.edges)
        avg_edges = num_edges / max(num_nodes, 1)

        self.allow_neurogenesis = num_nodes < self.max_nodes
        self.allow_synaptogenesis = avg_edges < self.max_edges_per_node

        return self.global_lr_multiplier
```

---

## 8. Multimodal I/O System

### 8.1 Input Encoders

```python
class TextEncoder:
    def __init__(self, vocab_size: int = 8192, embed_dim: int = 64):
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_encoding = nn.Embedding(512, embed_dim)
        self.tokenizer = load_or_train_tokenizer(vocab_size)

    def encode(self, text: str) -> List[torch.Tensor]:
        tokens = self.tokenizer.encode(text)
        positions = torch.arange(len(tokens))
        embeddings = self.embedding(torch.tensor(tokens)) + self.position_encoding(positions)
        return [emb for emb in embeddings]


class ImageEncoder:
    def __init__(self, patch_size: int = 16, embed_dim: int = 64):
        self.patch_proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def encode(self, image: torch.Tensor) -> List[torch.Tensor]:
        patches = self.patch_proj(image.unsqueeze(0))
        patches = patches.flatten(2).squeeze(0).T
        return [p for p in patches]
```

### 8.2 Output Decoders

```python
class TextDecoder:
    def __init__(self, vocab_size: int = 8192, embed_dim: int = 64):
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        self.tokenizer = load_or_train_tokenizer(vocab_size)

    def decode(self, activation: torch.Tensor) -> str:
        logits = self.output_proj(activation)
        token_id = logits.argmax().item()
        return self.tokenizer.decode([token_id])
```

### 8.3 Dataset Feeders

```python
class TextDatasetFeeder:
    def __init__(self, dataset_path: str, chunk_size: int = 32, overlap: int = 8):
        self.dataset = load_dataset(dataset_path, streaming=True)
        self.chunk_size = chunk_size

    def generate_experience(self):
        text = next(self.dataset)
        tokens = tokenize(text)
        split_point = len(tokens) // 2
        context_tokens = tokens[:split_point]
        target_tokens = tokens[split_point:split_point + self.chunk_size]
        context_embedding = self.text_encoder.encode(context_tokens)
        target_embedding = self.text_encoder.encode(target_tokens)
        return (
            {"text": torch.stack(context_embedding)},
            torch.stack(target_embedding)
        )

class MultimodalCurriculum:
    def __init__(self):
        self.schedule = [
            (0, 10000, {"text": 0.8, "image": 0.2}),
            (10000, 50000, {"text": 0.5, "image": 0.5}),
            (50000, None, {"text": 0.4, "image": 0.3, "interleaved": 0.3}),
        ]
```

---

## 9. Main Training Loop

```python
class SOMA:
    def __init__(self, config: SOMAConfig):
        self.graph = Graph()
        self.working_memory = WorkingMemory(config.wm_slots, config.wm_dim)
        self.episodic_memory = EpisodicMemory(config.episodic_capacity, config.key_dim, config.value_dim)
        self.curiosity = CuriosityModule(config.num_curiosity_domains)
        self.homeostasis = HomeostaticRegulator(config.max_nodes, config.max_edges_per_node)
        self.development = DevelopmentSchedule()
        self.global_step = 0
        self._initialize_seed_graph(config)

    def step(self, inputs, targets=None):
        """Process one interaction step."""
        # 1. Forward pass through graph
        outputs = execute_graph(self.graph, inputs)

        # 2. Working memory update
        if outputs:
            context = list(outputs.values())[0]
            wm_read = self.working_memory.read(context)
            self.working_memory.write(context, wm_read)
            self.working_memory.step()

        # 3. Compute loss
        loss = None
        prediction_error = 0.0
        if targets is not None and outputs:
            output_tensor = list(outputs.values())[0]
            loss = F.mse_loss(output_tensor, targets)
            prediction_error = loss.item()

        # 4. Encode experience to episodic memory
        if loss is not None:
            experience = self._create_experience_vector(inputs, outputs, targets)
            self.episodic_memory.encode(experience, prediction_error)

        # 5. Learning step
        if loss is not None:
            lr_multiplier = self.homeostasis.update(self.graph, loss.item())
            update_step(self.graph, loss, self.global_step)

        # 6. Curiosity tracking
        if outputs:
            input_repr = list(inputs.values())[0].mean(dim=0) if list(inputs.values())[0].dim() > 1 else list(inputs.values())[0]
            curiosity_score = self.curiosity.compute_curiosity(input_repr, prediction_error)

        # 7. Periodic structural updates
        if self.global_step % SYNAPTOGENESIS_INTERVAL == 0:
            if self.homeostasis.allow_synaptogenesis:
                synaptogenesis(self.graph, self._get_current_activations(), self.global_step)

        if self.global_step % NEUROGENESIS_INTERVAL == 0:
            if self.homeostasis.allow_neurogenesis:
                neurogenesis(self.graph, self._recent_errors, self.global_step)

        # 8. Consolidation (artificial sleep)
        if self.global_step % CONSOLIDATION_INTERVAL == 0:
            consolidation_cycle(self, num_replay_steps=CONSOLIDATION_REPLAY_STEPS)

        # 9. Advance global step
        self.global_step += 1
        return outputs, loss, prediction_error

    def interactive_session(self, text_input: str) -> str:
        """Convenience method for text-based interaction."""
        encoded = self.text_encoder.encode(text_input)
        input_dict = {"text": torch.stack(encoded)}
        output_tokens = []
        for _ in range(MAX_OUTPUT_TOKENS):
            outputs, _, _ = self.step(input_dict, targets=None)
            if "text" in outputs:
                token = self.text_decoder.decode(outputs["text"])
                output_tokens.append(token)
                if token == "<EOS>":
                    break
                input_dict = {"text": outputs["text"].unsqueeze(0)}
        return "".join(output_tokens)

    def save_state(self, path: str):
        """Save complete system state."""
        state = {
            "graph": self.graph.serialize(),
            "working_memory": self.working_memory.state_dict(),
            "episodic_memory": self.episodic_memory.state_dict(),
            "curiosity": self.curiosity.state_dict(),
            "homeostasis": self.homeostasis.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }
        torch.save(state, path)

    def load_state(self, path: str):
        """Load system state from checkpoint."""
        state = torch.load(path)
        self.graph = Graph.deserialize(state["graph"])
        self.working_memory.load_state_dict(state["working_memory"])
        self.episodic_memory.load_state_dict(state["episodic_memory"])
        self.curiosity.load_state_dict(state["curiosity"])
        self.homeostasis.load_state_dict(state["homeostasis"])
        self.global_step = state["global_step"]
```

---

## 10. Configuration and Hyperparameters

```python
@dataclass
class SOMAConfig:
    # Graph Dimensions
    sensor_output_dim: int = 64
    associator_input_dim: int = 64
    associator_hidden_dim: int = 128
    associator_output_dim: int = 64
    integrator_input_dim: int = 128
    integrator_hidden_dim: int = 256
    integrator_output_dim: int = 128
    position_dim: int = 16

    # Memory Systems
    wm_slots: int = 32
    wm_dim: int = 128
    wm_decay_rate: float = 0.95
    episodic_capacity: int = 10000
    key_dim: int = 128
    value_dim: int = 256

    # I/O
    input_modalities: List[str] = field(default_factory=lambda: ["text"])
    output_modalities: List[str] = field(default_factory=lambda: ["text"])
    vocab_size: int = 8192
    text_embed_dim: int = 64

    # Growth
    initial_associator_count: int = 32
    initial_integrator_count: int = 8
    max_nodes: int = 50000
    max_edges_per_node: float = 20.0

    # Learning
    base_lr: float = 0.001
    youth_lr_multiplier: float = 3.0
    hebbian_lr: float = 0.0001
    consolidation_lr_ratio: float = 0.1
    maturity_increment: float = 0.0001

    # Growth Thresholds
    activation_threshold: float = 0.1
    synaptogenesis_rate: float = 0.01
    neurogenesis_threshold: float = 1.2
    edge_strength_threshold: float = 0.001
    inactivity_threshold: int = 5000
    pruning_grace_period: int = 2000
    myelination_strength_threshold: float = 0.5
    myelination_age_threshold: int = 10000
    max_edge_weight: float = 5.0
    locality_scale: float = 2.0
    position_jitter: float = 0.1

    # Intervals
    synaptogenesis_interval: int = 100
    neurogenesis_interval: int = 500
    consolidation_interval: int = 1000
    consolidation_replay_steps: int = 100
    consolidation_error_threshold: float = 0.5  # Error threshold to trigger neurogenesis during consolidation
    pruning_interval: int = 1000
    checkpoint_interval: int = 5000

    # Curiosity
    num_curiosity_domains: int = 8

    # Homeostasis
    default_target_activation: float = 0.5
    gain_min: float = 0.1
    gain_max: float = 10.0

    # Sequence Processing
    max_output_tokens: int = 128
    max_input_tokens: int = 256
```

---

## 11. Resource Estimation

### Memory Budget (RTX 3090, 24GB VRAM)

| Component | Estimated VRAM at Scale |
|-----------|------------------------|
| Graph nodes (50K nodes, ~128 params each) | ~400 MB |
| Graph edges (500K edges, projections) | ~2 GB |
| Working memory (32 slots x 128 dim) | ~16 KB |
| Episodic memory (10K episodes x 256 dim) | ~10 MB |
| I/O encoders/decoders | ~100 MB |
| Activations / gradients (during forward/backward) | ~2-4 GB |
| PyTorch overhead | ~2 GB |
| **Total estimated** | **~6-8 GB** |

---

## 12. Key Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Graph execution becomes slow as it grows | Activation sparsity; PyTorch sparse ops; MAX_NODES cap |
| Catastrophic forgetting during consolidation | Smaller LR; surprise-prioritized replay; test at every stage |
| Graph collapses to trivial topology | Homeostatic regulation; curiosity drives exploration; diversity pressure |
| Unbounded growth exhausts memory | Hard caps on nodes/edges; aggressive pruning; myelination reduces count |
| No convergence on any meaningful task | Start with simple synthetic tasks; validate each stage independently |
| Cyclic graph causes infinite loops | Back-edges use previous timestep activations; max propagation depth |
