"""Stage 5 integration test — cross-modal association.

Exercises the multimodal I/O stack:
- ``ImageEncoder`` + ``TextEncoder`` share a dim with their sensor nodes.
- ``MultimodalCurriculum`` drives modality-weighted sampling.
- A single associator receives both text and image signals; the graph
  learns to associate paired inputs with a shared latent output.

Asserts:
- No NaNs for the full run.
- Loss decreases when paired signals repeat.
- Dropping one modality changes the output (both modalities matter).
- Curriculum sampling honors the schedule weights.
"""

from __future__ import annotations

import itertools

import pytest
import torch
from torch.nn import functional as F  # noqa: N812

from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.execution import execute_graph
from soma.core.graph import Graph
from soma.core.learning import update_step
from soma.core.node import Node, NodeType
from soma.io.image_encoder import ImageEncoder
from soma.io.multimodal_curriculum import CurriculumWindow, MultimodalCurriculum
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer


@pytest.fixture
def config() -> SOMAConfig:
    return SOMAConfig(
        base_lr=0.03,
        youth_lr_multiplier=1.0,
        hebbian_lr=0.0001,
        activation_threshold=0.01,
        max_edge_weight=3.0,
        text_embed_dim=8,
        sensor_output_dim=8,
        associator_input_dim=8,
        associator_output_dim=8,
    )


def _build_crossmodal_graph(config: SOMAConfig, dim: int = 8) -> Graph:
    graph = Graph()
    text_sensor = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
    image_sensor = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
    assoc = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
    out = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
    graph.add_node(text_sensor, modality="text")
    graph.add_node(image_sensor, modality="image")
    graph.add_node(assoc)
    graph.add_node(out, modality="text")

    def _edge(src: Node, tgt: Node, *, weight: float = 1.0) -> Edge:
        return Edge(
            source_id=src.id,
            target_id=tgt.id,
            source_output_dim=src.output_dim,
            target_input_dim=tgt.input_dim,
            creation_step=0,
            initial_weight=weight,
        )

    graph.add_edge(_edge(text_sensor, assoc, weight=0.5))
    graph.add_edge(_edge(image_sensor, assoc, weight=0.5))
    graph.add_edge(_edge(assoc, out))
    return graph


class TestCrossModalAssociation:
    def test_outputs_depend_on_both_modalities(self, config: SOMAConfig) -> None:
        """Zeroing one modality should alter the output."""
        torch.manual_seed(0)
        graph = _build_crossmodal_graph(config)
        dim = 8

        text = torch.randn(dim)
        image = torch.randn(dim)
        both_outputs, _ = execute_graph(
            graph, inputs={"text": text, "image": image}, current_step=0
        )
        text_only_outputs, _ = execute_graph(
            graph, inputs={"text": text, "image": torch.zeros(dim)}, current_step=1
        )
        # Both runs produce a text output.
        assert "text" in both_outputs
        assert "text" in text_only_outputs
        # Turning image off must change the output (image edge feeds assoc).
        assert not torch.allclose(both_outputs["text"], text_only_outputs["text"], atol=1e-4)

    def test_learns_paired_mapping(self, config: SOMAConfig) -> None:
        """Feed the same paired signals repeatedly; loss should decrease."""
        torch.manual_seed(0)
        graph = _build_crossmodal_graph(config)
        dim = 8
        rng = torch.Generator().manual_seed(1)

        # Fixed paired samples: each "concept" has a text embedding and an
        # image embedding; target for the output is the sum (a simple
        # cross-modal composition).
        num_concepts = 5
        text_vecs = [torch.randn(dim, generator=rng) for _ in range(num_concepts)]
        image_vecs = [torch.randn(dim, generator=rng) for _ in range(num_concepts)]
        targets = [(t + i) * 0.5 for t, i in zip(text_vecs, image_vecs, strict=True)]

        losses: list[float] = []
        for step in range(200):
            idx = step % num_concepts
            outputs, activations = execute_graph(
                graph,
                inputs={"text": text_vecs[idx].clone(), "image": image_vecs[idx].clone()},
                current_step=step,
            )
            loss = F.mse_loss(outputs["text"], targets[idx])
            losses.append(float(loss.item()))
            update_step(graph, loss, activations, config)

        early = sum(losses[:20]) / 20
        late = sum(losses[-20:]) / 20
        assert late < early

    def test_curriculum_sampling_honors_schedule(self) -> None:
        """Text-heavy schedule should give far more text samples than image."""
        curriculum = MultimodalCurriculum([CurriculumWindow(0, None, {"text": 0.9, "image": 0.1})])
        rng = torch.Generator().manual_seed(0)
        samples = [curriculum.sample_modality(step=0, rng=rng) for _ in range(1000)]
        text_count = samples.count("text")
        image_count = samples.count("image")
        assert text_count > image_count * 4  # roughly respects 9:1 ratio


class TestEncoderIntegration:
    @pytest.fixture(scope="class")
    def tokenizer(self) -> object:
        corpus = [
            "red circle blue square green triangle " * 5,
            "cat dog bird fish " * 5,
            "running jumping swimming flying " * 5,
        ]
        return train_bpe_tokenizer(corpus, vocab_size=64)

    def test_encoders_produce_compatible_shapes(
        self,
        config: SOMAConfig,
        tokenizer: object,
    ) -> None:
        dim = config.sensor_output_dim
        text_encoder = TextEncoder(tokenizer, embed_dim=dim, max_seq_len=32)
        image_encoder = ImageEncoder(patch_size=4, embed_dim=dim, in_channels=3)
        tokens = text_encoder.encode("red circle")
        patches = image_encoder.encode(torch.randn(3, 8, 8))
        assert all(t.shape == (dim,) for t in tokens)
        assert all(p.shape == (dim,) for p in patches)

    def test_end_to_end_two_modalities_no_nans(
        self,
        config: SOMAConfig,
        tokenizer: object,
    ) -> None:
        torch.manual_seed(0)
        dim = config.sensor_output_dim
        text_encoder = TextEncoder(tokenizer, embed_dim=dim, max_seq_len=32)
        image_encoder = ImageEncoder(patch_size=4, embed_dim=dim, in_channels=3)
        graph = _build_crossmodal_graph(config, dim=dim)

        samples = [
            ("red circle", torch.randn(3, 8, 8)),
            ("cat dog", torch.randn(3, 8, 8)),
        ]
        iterator = itertools.cycle(samples)
        for step in range(20):
            text, image = next(iterator)
            text_vec = text_encoder.encode_batch(text)[0]  # first token's embedding
            image_vec = image_encoder.encode_batch(image)[0]  # first patch embedding
            outputs, _ = execute_graph(
                graph,
                inputs={"text": text_vec, "image": image_vec},
                current_step=step,
            )
            assert "text" in outputs
            assert torch.isfinite(outputs["text"]).all()
