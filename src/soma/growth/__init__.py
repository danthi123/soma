"""SOMA growth package: synaptogenesis, neurogenesis, pruning, myelination."""

from soma.growth.pruning import PruningResult, pruning
from soma.growth.synaptogenesis import synaptogenesis

__all__ = ["PruningResult", "pruning", "synaptogenesis"]
