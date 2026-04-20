# Concrete examples from each retrieval partition

## (0,1): only SOMA retrieves gold — top 5 F1 deltas

- **chroma_cosine**: `118b2229` (single-session-user)
  - hypothesis: `I don't know`
  - gold answer: `45 minutes each way`
  - F1=0.000 | R@5_hit=0

- **soma_hybrid**: `118b2229` (single-session-user)
  - hypothesis: `45 minutes each way`
  - gold answer: `45 minutes each way`
  - F1=1.000 | R@5_hit=1

  - delta F1 = +1.000

- **chroma_cosine**: `58bf7951` (single-session-user)
  - hypothesis: `I don't know`
  - gold answer: `The Glass Menagerie`
  - F1=0.000 | R@5_hit=0

- **soma_hybrid**: `58bf7951` (single-session-user)
  - hypothesis: `The Glass Menagerie`
  - gold answer: `The Glass Menagerie`
  - F1=1.000 | R@5_hit=1

  - delta F1 = +1.000

- **chroma_cosine**: `1e043500` (single-session-user)
  - hypothesis: `I don't know`
  - gold answer: `Summer Vibes`
  - F1=0.000 | R@5_hit=0

- **soma_hybrid**: `1e043500` (single-session-user)
  - hypothesis: `Summer Vibes`
  - gold answer: `Summer Vibes`
  - F1=1.000 | R@5_hit=1

  - delta F1 = +1.000

- **chroma_cosine**: `a06e4cfe` (single-session-user)
  - hypothesis: `I don't know`
  - gold answer: `3:1`
  - F1=0.000 | R@5_hit=0

- **soma_hybrid**: `a06e4cfe` (single-session-user)
  - hypothesis: `3:1`
  - gold answer: `3:1`
  - F1=1.000 | R@5_hit=1

  - delta F1 = +1.000

- **chroma_cosine**: `f4f1d8a4` (single-session-user)
  - hypothesis: `I don't know`
  - gold answer: `my sister`
  - F1=0.000 | R@5_hit=0

- **soma_hybrid**: `f4f1d8a4` (single-session-user)
  - hypothesis: `My sister`
  - gold answer: `my sister`
  - F1=1.000 | R@5_hit=1

  - delta F1 = +1.000

## (1,1) both retrieve gold, SOMA wins F1 — top 5 deltas

- **chroma_cosine**: `ad7109d1` (single-session-user)
  - hypothesis: `I don't know`
  - gold answer: `500 Mbps`
  - F1=0.000 | R@5_hit=1

- **soma_hybrid**: `ad7109d1` (single-session-user)
  - hypothesis: `500 Mbps`
  - gold answer: `500 Mbps`
  - F1=1.000 | R@5_hit=1

  - delta F1 = +1.000

- **chroma_cosine**: `853b0a1d` (single-session-user)
  - hypothesis: `I don't know`
  - gold answer: `18`
  - F1=0.000 | R@5_hit=1

- **soma_hybrid**: `853b0a1d` (single-session-user)
  - hypothesis: `18`
  - gold answer: `18`
  - F1=1.000 | R@5_hit=1

  - delta F1 = +1.000

- **chroma_cosine**: `8550ddae` (single-session-user)
  - hypothesis: `I don't know`
  - gold answer: `lavender gin fizz`
  - F1=0.000 | R@5_hit=1

- **soma_hybrid**: `8550ddae` (single-session-user)
  - hypothesis: `Lavender gin fizz`
  - gold answer: `lavender gin fizz`
  - F1=1.000 | R@5_hit=1

  - delta F1 = +1.000

- **chroma_cosine**: `60d45044` (single-session-user)
  - hypothesis: `I don't know`
  - gold answer: `Japanese short-grain rice`
  - F1=0.000 | R@5_hit=1

- **soma_hybrid**: `60d45044` (single-session-user)
  - hypothesis: `Japanese short-grain rice`
  - gold answer: `Japanese short-grain rice`
  - F1=1.000 | R@5_hit=1

  - delta F1 = +1.000

- **chroma_cosine**: `86b68151` (single-session-user)
  - hypothesis: `I don't know`
  - gold answer: `IKEA`
  - F1=0.000 | R@5_hit=1

- **soma_hybrid**: `86b68151` (single-session-user)
  - hypothesis: `IKEA`
  - gold answer: `IKEA`
  - F1=1.000 | R@5_hit=1

  - delta F1 = +1.000

## (1,1) both retrieve gold, chroma wins F1 — top 5 deltas

- **chroma_cosine**: `00ca467f` (multi-session)
  - hypothesis: `2`
  - gold answer: `2`
  - F1=1.000 | R@5_hit=1

- **soma_hybrid**: `00ca467f` (multi-session)
  - hypothesis: `3`
  - gold answer: `2`
  - F1=0.000 | R@5_hit=1

  - delta F1 = -1.000

- **chroma_cosine**: `87f22b4a` (multi-session)
  - hypothesis: `$120`
  - gold answer: `$120`
  - F1=1.000 | R@5_hit=1

- **soma_hybrid**: `87f22b4a` (multi-session)
  - hypothesis: `I don't know`
  - gold answer: `$120`
  - F1=0.000 | R@5_hit=1

  - delta F1 = -1.000

- **chroma_cosine**: `gpt4_70e84552` (temporal-reasoning)
  - hypothesis: `Fixing the fence`
  - gold answer: `Fixing the fence`
  - F1=1.000 | R@5_hit=1

- **soma_hybrid**: `gpt4_70e84552` (temporal-reasoning)
  - hypothesis: `Trimming goats' hooves`
  - gold answer: `Fixing the fence`
  - F1=0.000 | R@5_hit=1

  - delta F1 = -1.000

- **chroma_cosine**: `9ea5eabc` (knowledge-update)
  - hypothesis: `Paris`
  - gold answer: `Paris`
  - F1=1.000 | R@5_hit=1

- **soma_hybrid**: `9ea5eabc` (knowledge-update)
  - hypothesis: `Hawaii`
  - gold answer: `Paris`
  - F1=0.000 | R@5_hit=1

  - delta F1 = -1.000

- **chroma_cosine**: `06db6396` (knowledge-update)
  - hypothesis: `5`
  - gold answer: `5`
  - F1=1.000 | R@5_hit=1

- **soma_hybrid**: `06db6396` (knowledge-update)
  - hypothesis: `4`
  - gold answer: `5`
  - F1=0.000 | R@5_hit=1

  - delta F1 = -1.000

## (1,0): only chroma retrieves gold

- **chroma_cosine**: `gpt4_f2262a51` (multi-session)
  - hypothesis: `2`
  - gold answer: `I visited three different doctors: a primary care physician, an ENT specialist, and a dermatologist.`
  - F1=0.000 | R@5_hit=1

- **soma_hybrid**: `gpt4_f2262a51` (multi-session)
  - hypothesis: `I don't know`
  - gold answer: `I visited three different doctors: a primary care physician, an ENT specialist, and a dermatologist.`
  - F1=0.111 | R@5_hit=0

  - delta F1 = +0.111

- **chroma_cosine**: `06f04340` (single-session-preference)
  - hypothesis: `I don't know`
  - gold answer: `The user would prefer dinner suggestions that incorporate their homegrown cherry tomatoes and herbs like basil and mint,`
  - F1=0.000 | R@5_hit=1

- **soma_hybrid**: `06f04340` (single-session-preference)
  - hypothesis: `I don't know`
  - gold answer: `The user would prefer dinner suggestions that incorporate their homegrown cherry tomatoes and herbs like basil and mint,`
  - F1=0.000 | R@5_hit=0

  - delta F1 = +0.000

- **chroma_cosine**: `09d032c9` (single-session-preference)
  - hypothesis: `I don't know`
  - gold answer: `The user would prefer responses that build upon their previous mention of purchasing a portable power bank, such as sugg`
  - F1=0.000 | R@5_hit=1

- **soma_hybrid**: `09d032c9` (single-session-preference)
  - hypothesis: `I don't know`
  - gold answer: `The user would prefer responses that build upon their previous mention of purchasing a portable power bank, such as sugg`
  - F1=0.000 | R@5_hit=0

  - delta F1 = +0.000

- **chroma_cosine**: `8e91e7d9` (multi-session)
  - hypothesis: `1`
  - gold answer: `4`
  - F1=0.000 | R@5_hit=1

- **soma_hybrid**: `8e91e7d9` (multi-session)
  - hypothesis: `I don't know`
  - gold answer: `4`
  - F1=0.000 | R@5_hit=0

  - delta F1 = +0.000

- **chroma_cosine**: `gpt4_e061b84g` (temporal-reasoning)
  - hypothesis: `I don't know`
  - gold answer: `The company's annual charity soccer tournament.`
  - F1=0.000 | R@5_hit=1

- **soma_hybrid**: `gpt4_e061b84g` (temporal-reasoning)
  - hypothesis: `I don't know`
  - gold answer: `The company's annual charity soccer tournament.`
  - F1=0.000 | R@5_hit=0

  - delta F1 = +0.000

- **chroma_cosine**: `gpt4_4929293b` (temporal-reasoning)
  - hypothesis: `Michael's engagement party`
  - gold answer: `my cousin's wedding`
  - F1=0.000 | R@5_hit=1

- **soma_hybrid**: `gpt4_4929293b` (temporal-reasoning)
  - hypothesis: `I don't know`
  - gold answer: `my cousin's wedding`
  - F1=0.000 | R@5_hit=0

  - delta F1 = +0.000

