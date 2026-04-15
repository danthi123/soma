# Verbalizer Sample Comparison (Fresh vs Trained)

**LLM**: `Qwen/Qwen2.5-3B-Instruct` (fp16)
**SOMA bundle**: `checkpoints\bootstrap-src-fresh-2026-04-15`
**Trained verbalizer**: `artifacts\bootstrap-2026-04-15-qwen3b-wikitext2\verbalizer_step_15500`
**max_new_tokens**: 40, do_sample=False (greedy)

Both runs use the SAME SOMA state per prompt (each session feeds
the prompt through SOMA, so SOMA-state evolution is identical).
The only varying factor is the verbalizer's projector weights:
fresh = randomly-initialised, trained = step_15500 (best held-out).

## Prompt: `The history of`

**Fresh** (untrained verbalizer):
```
 the development of the theory of the firm is a history of the struggle between two opposing views: the view that the firm is an economic unit, and the view that the firm is a political unit.
```

**Trained** (step_15500 verbalizer):
```
 the band is a story of constant change and evolution . The band has gone through many lineup changes over the years , with the current lineup consisting of guitarist and vocalist John Cale , bassist and vocalist
```

## Prompt: `In a typical wikipedia article,`

**Fresh** (untrained verbalizer):
```
 there are many references to other articles. For ease of understanding and navigation, it would be helpful if we can create a reference diagram that shows the relationships between articles. Your task is to identify pairs of
```

**Trained** (step_15500 verbalizer):
```
 the first paragraph is a summary of the article 's content , and the second paragraph is a list of the article 's main points . The third paragraph is a list of the article 's references
```

## Prompt: `The capital city of France`

**Fresh** (untrained verbalizer):
```
 is Paris. The capital city of France is Paris. 选项： A. 正确 B. 错误
A. 正确

以下是中国关于大学考试的主观题，请
```

**Trained** (step_15500 verbalizer):
```
 , Paris , is the most visited city in the world . The city has a population of 2.2 million people within its administrative limits and 11 million in its metropolitan area . It is
```

## Prompt: `Quantum mechanics describes`

**Fresh** (untrained verbalizer):
```
 the behavior of particles at the atomic and subatomic level. It is a fundamental theory in physics that explains the physical properties of nature at the smallest scales. The key concepts in quantum mechanics include:

1
```

**Trained** (step_15500 verbalizer):
```
 the behavior of matter and energy at the atomic and sub @-@ atomic level . It is a fundamental theory in physics , which describes nature at the smallest scales of energy levels of atoms and subatomic
```

## Prompt: `After the war ended,`

**Fresh** (untrained verbalizer):
```
 the Japanese government began to implement a series of policies to rebuild and develop the country. One of these policies was the "Nationalization of Land" policy, which aimed to consolidate land ownership and promote agricultural
```

**Trained** (step_15500 verbalizer):
```
 the 1946 film The Great Dictator was released , in which Chaplin played a Jewish barber who becomes dictator of an invented country called Tomainia . In the film , Chaplin
```

