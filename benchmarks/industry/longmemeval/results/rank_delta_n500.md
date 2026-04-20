# Rank-delta: chroma vs SOMA  |  shared N=500

## Partition

| Cell | N | % |
| --- | ---: | ---: |
| both retrieve, same rank | 385 | 77.0% |
| SOMA places gold at lower (better) rank | 61 | 12.2% |
| SOMA places gold at higher (worse) rank | 14 | 2.8% |
| SOMA rescues (chroma miss, SOMA hit) | 30 | 6.0% |
| SOMA loses (chroma hit, SOMA miss) | 6 | 1.2% |
| both miss | 4 | 0.8% |

## Pair distribution (chroma_rank, soma_rank)

Rows = chroma rank, columns = SOMA rank. 0 = missed.

| chroma \ soma | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rank 0 | 4 | 14 | 9 | 2 | 3 | 2 |
| rank 1 | 1 | 376 | 9 | 0 | 1 | 1 |
| rank 2 | 4 | 27 | 8 | 1 | 0 | 0 |
| rank 3 | 1 | 12 | 0 | 0 | 0 | 2 |
| rank 4 | 0 | 9 | 4 | 0 | 1 | 0 |
| rank 5 | 0 | 5 | 1 | 2 | 1 | 0 |

Mean chroma rank (hits only): 1.318  (N=466)
Mean SOMA rank (hits only):   1.161  (N=490)
