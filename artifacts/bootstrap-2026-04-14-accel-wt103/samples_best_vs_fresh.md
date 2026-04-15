# Verbalizer Sample Comparison (Fresh vs Trained)

**LLM**: `Qwen/Qwen2.5-3B-Instruct` (fp16)
**SOMA bundle**: `checkpoints\bootstrap-src-fresh-2026-04-15`
**Trained verbalizer**: `artifacts\bootstrap-2026-04-14-accel-wt103\verbalizer_best`
**max_new_tokens**: 40, do_sample=False (greedy)

Both runs use the SAME SOMA state per prompt (each session feeds
the prompt through SOMA, so SOMA-state evolution is identical).
The only varying factor is the verbalizer's projector weights:
fresh = randomly-initialised, trained = step_15500 (best held-out).

## Prompt: `The history of`

**Fresh** (untrained verbalizer):
```
 the Internet is a story of many people working together to create something that has become an integral part of our lives. The Internet was not created by one person or one company, but rather by a group
```

**Trained** (step_15500 verbalizer):
```
 the film was not known to the cast and crew , who were told that it was a " low budget " production . The film 's budget was $ 10 @.@ 5 million ,
```

## Prompt: `In a typical wikipedia article,`

**Fresh** (untrained verbalizer):
```
 the first sentence is a summary of the article. This sentence contains the key information and can be used to understand the main idea of the article. However, in the case of the article on the Chinese
```

**Trained** (step_15500 verbalizer):
```
 the first sentence is a summary of the article 's content . The first sentence of this article is : " The 1984 @-@ 85 NBA season was the 3
```

## Prompt: `The capital city of France`

**Fresh** (untrained verbalizer):
```
 is Paris. The capital city of Germany is Berlin. The capital city of Italy is Rome. The capital city of Spain is Madrid. The capital city of Portugal is Lisbon. The capital city of Greece
```

**Trained** (step_15500 verbalizer):
```
 , Paris , is the setting for the film . The city 's landmarks are used as backdrops in the film , including the Eiffel Tower and the Louvre . 

 = = Plot =
```

## Prompt: `Quantum mechanics describes`

**Fresh** (untrained verbalizer):
```
 the behavior of particles at the quantum scale. It is a fundamental theory in physics that explains the physical properties of matter and energy at the smallest scales, such as atoms and subatomic particles. Quantum mechanics
```

**Trained** (step_15500 verbalizer):
```
 the behavior of particles at the atomic and subatomic level . It is a fundamental theory in physics , which describes how matter and its motion are affected by the fundamental forces acting between particles . Quantum mechanics i
```

## Prompt: `After the war ended,`

**Fresh** (untrained verbalizer):
```
 the United States and the Soviet Union became the world's two superpowers. The Cold War, a period of tension and hostility between the two countries, lasted for decades. During this time, both nations
```

**Trained** (step_15500 verbalizer):
```
 the British government decided to use the captured German aircraft as a means of training pilots . The first batch of aircraft was sent to the United States for testing and evaluation , but the majority were sent to the
```

