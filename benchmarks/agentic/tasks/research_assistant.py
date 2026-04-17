"""Research assistant with iterative refinement (50-100 turns).

The agent must answer 5 research questions by searching a simulated
knowledge base of 20+ papers, some with contradictory findings.
Later evidence may require revising earlier findings.  Tests SOMA's
ability to retain paper contents and track evolving conclusions.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from benchmarks.agentic.metrics import TaskResult

# ---------------------------------------------------------------------------
# Paper databases (one per scenario variant)
# ---------------------------------------------------------------------------
_SCENARIOS: list[dict] = [
    # Scenario 0: AI model scaling research
    {
        "papers": [
            {
                "id": "P001",
                "title": "Scaling Laws for Neural Language Models",
                "abstract": (
                    "We study empirical scaling laws for language model "
                    "performance. Loss scales as a power-law with model "
                    "size, dataset size, and compute. Larger models are "
                    "more sample-efficient."
                ),
                "full_text": (
                    "Our experiments across 7 orders of magnitude show "
                    "that cross-entropy loss follows L(N) = aN^(-0.076) "
                    "where N is parameter count. Data requirements grow "
                    "sub-linearly with model size. We recommend scaling "
                    "model size faster than dataset size for optimal "
                    "compute allocation."
                ),
                "findings": ["power-law scaling", "larger models sample-efficient"],
                "tags": ["scaling", "efficiency", "language-models"],
            },
            {
                "id": "P002",
                "title": "Training Compute-Optimal Language Models",
                "abstract": (
                    "Current large language models are significantly "
                    "undertrained. For compute-optimal training, model "
                    "size and training data should be scaled equally."
                ),
                "full_text": (
                    "Contrary to prior scaling recommendations, we find "
                    "that model size and training tokens should scale "
                    "roughly equally. A 70B model trained on 1.4T tokens "
                    "outperforms a 280B model trained on 300B tokens "
                    "with the same compute budget. This contradicts "
                    "the recommendation to scale model size faster."
                ),
                "findings": [
                    "scale data equally with model size",
                    "contradicts P001 scaling recommendation",
                ],
                "tags": ["scaling", "compute-optimal", "data"],
            },
            {
                "id": "P003",
                "title": "Emergent Abilities of Large Language Models",
                "abstract": (
                    "We document abilities that emerge unpredictably "
                    "as models scale. These include chain-of-thought "
                    "reasoning, which appears around 100B parameters."
                ),
                "full_text": (
                    "Analysis of 200+ benchmarks reveals discontinuous "
                    "jumps in capability at specific scale thresholds. "
                    "Chain-of-thought reasoning emerges at ~100B params. "
                    "Few-shot translation at ~10B. Code generation "
                    "improves sharply at ~50B. These emergent abilities "
                    "cannot be predicted from smaller-scale experiments."
                ),
                "findings": [
                    "emergent abilities at scale thresholds",
                    "chain-of-thought at 100B",
                    "unpredictable from small models",
                ],
                "tags": ["emergence", "scaling", "reasoning"],
            },
            {
                "id": "P004",
                "title": "Are Emergent Abilities a Mirage?",
                "abstract": (
                    "We challenge the claim of emergent abilities. "
                    "Apparent emergence is an artifact of discontinuous "
                    "metrics, not fundamental phase transitions."
                ),
                "full_text": (
                    "Re-analyzing the benchmarks from prior work using "
                    "continuous metrics (token-level log-likelihood) "
                    "instead of discontinuous ones (exact match), we "
                    "show smooth, predictable improvement. The 'emergent "
                    "abilities' disappear when measured properly. This "
                    "directly contradicts claims of unpredictable jumps "
                    "in P003."
                ),
                "findings": [
                    "emergence is metric artifact",
                    "smooth scaling with continuous metrics",
                    "contradicts P003 emergence claims",
                ],
                "tags": ["emergence", "metrics", "scaling"],
            },
            {
                "id": "P005",
                "title": "RLHF: Training Language Models to Follow Instructions",
                "abstract": (
                    "We fine-tune language models using reinforcement "
                    "learning from human feedback. A 1.3B RLHF model "
                    "is preferred over a 175B base model."
                ),
                "full_text": (
                    "InstructGPT shows that alignment training with RLHF "
                    "can make a 1.3B model outperform a 175B base model "
                    "on human preference. This suggests alignment may be "
                    "more important than raw scale for usefulness. RLHF "
                    "training cost is <2% of pretraining compute."
                ),
                "findings": [
                    "RLHF small > base large on preference",
                    "alignment more important than scale",
                ],
                "tags": ["rlhf", "alignment", "instruction-following"],
            },
            {
                "id": "P006",
                "title": "Mixture of Experts for Efficient Scaling",
                "abstract": (
                    "Sparse mixture-of-experts models achieve better "
                    "performance per FLOP than dense models by "
                    "activating only a subset of parameters per input."
                ),
                "full_text": (
                    "Our MoE architecture uses 64 experts with top-2 "
                    "routing. A 1.2T parameter MoE model uses the same "
                    "FLOPs as a 125B dense model but matches a 540B "
                    "dense model's quality. This suggests scaling "
                    "parameter count is beneficial even when compute "
                    "is fixed, supporting P001's emphasis on model size "
                    "but through sparse activation."
                ),
                "findings": [
                    "MoE matches larger dense at same FLOPs",
                    "sparse scaling is compute-efficient",
                ],
                "tags": ["moe", "efficiency", "scaling"],
            },
            {
                "id": "P007",
                "title": "Small Models Can Be Strong Reasoners",
                "abstract": (
                    "With careful distillation and chain-of-thought "
                    "fine-tuning, 7B models match 70B models on "
                    "reasoning benchmarks."
                ),
                "full_text": (
                    "We distill chain-of-thought reasoning from a 70B "
                    "teacher into a 7B student. The student matches the "
                    "teacher on GSM8K (78% vs 80%) and BBH (65% vs 68%). "
                    "This challenges the view that reasoning requires "
                    "large models and partially contradicts P003's "
                    "claim about reasoning emergence thresholds."
                ),
                "findings": [
                    "7B matches 70B on reasoning via distillation",
                    "challenges reasoning emergence threshold",
                ],
                "tags": ["distillation", "reasoning", "efficiency"],
            },
            {
                "id": "P008",
                "title": "Constitutional AI: Harmlessness from AI Feedback",
                "abstract": (
                    "We train AI assistants using AI-generated feedback "
                    "instead of human labels, reducing reliance on "
                    "human annotators while improving safety."
                ),
                "full_text": (
                    "Constitutional AI (CAI) uses a set of principles "
                    "to guide self-critique and revision. CAI models "
                    "are less harmful than RLHF models while being more "
                    "helpful. This extends P005's RLHF approach by "
                    "replacing human feedback with AI feedback guided "
                    "by explicit principles."
                ),
                "findings": [
                    "AI feedback can replace human feedback",
                    "CAI less harmful than RLHF",
                ],
                "tags": ["alignment", "safety", "constitutional-ai"],
            },
            {
                "id": "P009",
                "title": "DPO: Direct Preference Optimization",
                "abstract": (
                    "We show that the RL objective in RLHF can be "
                    "solved directly without a reward model, "
                    "simplifying alignment training."
                ),
                "full_text": (
                    "DPO reparameterizes the RLHF objective to avoid "
                    "training a separate reward model. DPO matches or "
                    "exceeds RLHF on summarization and dialogue while "
                    "using 50% less compute. This challenges P005's "
                    "claim that RLHF is the optimal alignment method "
                    "by showing a simpler alternative."
                ),
                "findings": [
                    "DPO matches RLHF without reward model",
                    "50% less compute than RLHF",
                ],
                "tags": ["alignment", "dpo", "efficiency"],
            },
            {
                "id": "P010",
                "title": "Retrieval-Augmented Generation for Knowledge Tasks",
                "abstract": (
                    "Augmenting language models with retrieval reduces "
                    "hallucination and improves factual accuracy without "
                    "increasing model size."
                ),
                "full_text": (
                    "RAG combines a retriever with a generator. A 7B "
                    "RAG model outperforms a 175B model without retrieval "
                    "on knowledge-intensive tasks (NQ, TriviaQA). This "
                    "provides an alternative to scaling: keep the model "
                    "small but augment with retrieval, contradicting "
                    "P001's emphasis on scaling model parameters."
                ),
                "findings": [
                    "RAG small > large without retrieval on knowledge tasks",
                    "alternative to parameter scaling",
                ],
                "tags": ["rag", "retrieval", "knowledge"],
            },
            {
                "id": "P011",
                "title": "Tool-Augmented Language Models",
                "abstract": (
                    "Teaching LLMs to use external tools (calculators, "
                    "search, code interpreters) dramatically improves "
                    "performance on tasks requiring precise computation."
                ),
                "full_text": (
                    "Tool-use training enables a 7B model to match "
                    "175B models on math (via calculator), factual QA "
                    "(via search), and coding (via interpreter). Like "
                    "P010's RAG approach, this suggests augmentation "
                    "may be more effective than scaling for specific "
                    "task categories."
                ),
                "findings": [
                    "tool-use enables small models to match large",
                    "augmentation as alternative to scaling",
                ],
                "tags": ["tools", "augmentation", "efficiency"],
            },
            {
                "id": "P012",
                "title": "The Bitter Lesson Revisited",
                "abstract": (
                    "We revisit Sutton's Bitter Lesson and argue that "
                    "general methods leveraging compute always win over "
                    "domain-specific engineering in the long run."
                ),
                "full_text": (
                    "Historical analysis confirms that scaling general "
                    "methods (search, learning) consistently outperforms "
                    "hand-engineered features. P010 and P011's "
                    "augmentation approaches may be temporary solutions "
                    "that scaling will eventually subsume. The evidence "
                    "supports P001's scaling focus as the long-term "
                    "winning strategy."
                ),
                "findings": [
                    "general compute scaling wins long-term",
                    "augmentation may be temporary",
                ],
                "tags": ["scaling", "compute", "philosophy"],
            },
            {
                "id": "P013",
                "title": "Efficient Fine-Tuning with LoRA",
                "abstract": (
                    "Low-Rank Adaptation achieves comparable fine-tuning "
                    "quality to full fine-tuning with 10,000x fewer "
                    "trainable parameters."
                ),
                "full_text": (
                    "LoRA adds small rank-decomposition matrices to "
                    "attention layers. A LoRA-tuned 7B model matches "
                    "full fine-tuning on GLUE/SuperGLUE while training "
                    "only 0.01% of parameters. Combined with P007's "
                    "distillation results, this enables practical "
                    "deployment of capable small models."
                ),
                "findings": [
                    "LoRA matches full fine-tuning at 0.01% params",
                    "enables practical small model deployment",
                ],
                "tags": ["fine-tuning", "efficiency", "lora"],
            },
            {
                "id": "P014",
                "title": "Multimodal Foundation Models",
                "abstract": (
                    "Training on text, images, and audio jointly "
                    "produces models that transfer better across "
                    "modalities than single-modality specialists."
                ),
                "full_text": (
                    "Our multimodal model trained on 5B text-image "
                    "pairs shows positive transfer: text understanding "
                    "improves with visual grounding. This extends "
                    "P001's scaling laws to multimodal settings where "
                    "more diverse data (not just more text) improves "
                    "performance."
                ),
                "findings": [
                    "multimodal training improves text understanding",
                    "data diversity matters beyond quantity",
                ],
                "tags": ["multimodal", "transfer", "scaling"],
            },
            {
                "id": "P015",
                "title": "Long Context Window Models",
                "abstract": (
                    "We extend context windows to 1M tokens with "
                    "sparse attention. Models can now process entire "
                    "codebases without retrieval augmentation."
                ),
                "full_text": (
                    "Using ring attention and hierarchical memory, "
                    "we extend context to 1M tokens. On document QA, "
                    "long-context models match RAG systems (P010) while "
                    "being simpler to deploy. However, retrieval latency "
                    "is worse: 10s for 1M context vs 0.5s for RAG. "
                    "This partially contradicts P010 by showing context "
                    "can replace retrieval, but validates P010's "
                    "efficiency advantage."
                ),
                "findings": [
                    "1M context matches RAG quality",
                    "but 20x slower than RAG at inference",
                ],
                "tags": ["context-window", "attention", "efficiency"],
            },
            {
                "id": "P016",
                "title": "Test-Time Compute Scaling",
                "abstract": (
                    "Allocating more compute at inference (thinking "
                    "longer) improves reasoning more effectively than "
                    "scaling model parameters."
                ),
                "full_text": (
                    "Chain-of-thought prompting, self-consistency, and "
                    "tree search at inference improve a 7B model to "
                    "match a 70B model on math/reasoning tasks. This "
                    "supports P007's finding that small models can "
                    "reason well and contradicts P003's claim that "
                    "reasoning requires large models. The key insight: "
                    "test-time compute is more efficient than training "
                    "compute for reasoning."
                ),
                "findings": [
                    "test-time compute more efficient than scale for reasoning",
                    "supports small model reasoning capability",
                ],
                "tags": ["inference", "reasoning", "efficiency"],
            },
            {
                "id": "P017",
                "title": "Synthetic Data for Pretraining",
                "abstract": (
                    "Models trained on synthetic data generated by "
                    "larger models match real-data-trained models, "
                    "addressing data scarcity concerns."
                ),
                "full_text": (
                    "We generate 1T tokens of synthetic math and code "
                    "data using a 70B model. A 7B model trained on this "
                    "synthetic data matches one trained on real data. "
                    "This extends P002's data scaling findings: when "
                    "real data is exhausted, synthetic data can maintain "
                    "the scaling trajectory."
                ),
                "findings": [
                    "synthetic data matches real data quality",
                    "addresses data scarcity for scaling",
                ],
                "tags": ["synthetic-data", "scaling", "data"],
            },
            {
                "id": "P018",
                "title": "Model Merging Without Retraining",
                "abstract": (
                    "Averaging weights of models fine-tuned on different "
                    "tasks produces a single model that performs well "
                    "across all tasks without additional training."
                ),
                "full_text": (
                    "Task arithmetic and TIES merging combine "
                    "specialized LoRA adapters (P013) into a single "
                    "model. The merged model retains 95% of each "
                    "specialist's performance. This offers a practical "
                    "alternative to multi-task training and supports "
                    "P013's LoRA approach as a building block."
                ),
                "findings": [
                    "model merging retains 95% specialist performance",
                    "complementary to LoRA fine-tuning",
                ],
                "tags": ["merging", "fine-tuning", "efficiency"],
            },
            {
                "id": "P019",
                "title": "Safety Alignment Tax is Minimal",
                "abstract": (
                    "We show that RLHF and DPO alignment reduces "
                    "general capability by only 1-3%, contradicting "
                    "the 'alignment tax' concern."
                ),
                "full_text": (
                    "Comparing aligned (P005, P009) and unaligned models "
                    "on 50 benchmarks, capability degradation is 1-3% "
                    "on average. On instruction-following tasks, aligned "
                    "models are 30-50% better. The net effect of "
                    "alignment is strongly positive."
                ),
                "findings": [
                    "alignment tax is 1-3% on general benchmarks",
                    "aligned models 30-50% better on instruction tasks",
                ],
                "tags": ["alignment", "safety", "evaluation"],
            },
            {
                "id": "P020",
                "title": "Scaling Laws for Data-Constrained Regimes",
                "abstract": (
                    "When data is limited, repeating data up to 4 "
                    "epochs maintains scaling benefits. Beyond that, "
                    "returns diminish sharply."
                ),
                "full_text": (
                    "We train models from 1B to 100B on datasets from "
                    "100B to 1T tokens with 1-16 epochs. Up to 4 epochs "
                    "of repetition, scaling laws from P001 and P002 "
                    "hold. Beyond 4 epochs, overfitting dominates and "
                    "scaling benefits plateau. This refines P002's "
                    "equal scaling recommendation for data-scarce "
                    "regimes."
                ),
                "findings": [
                    "data repetition up to 4 epochs maintains scaling",
                    "beyond 4 epochs scaling breaks down",
                ],
                "tags": ["scaling", "data", "overfitting"],
            },
        ],
        "questions": [
            {
                "question": (
                    "What is the consensus on the optimal ratio of "
                    "model size to training data for scaling?"
                ),
                "expected_claim": (
                    "P001 recommends scaling model size faster, but P002 "
                    "contradicts this saying they should scale equally."
                ),
                "required_papers": ["P001", "P002"],
                "keywords": [
                    "power-law",
                    "equally",
                    "compute-optimal",
                    "contradict",
                ],
            },
            {
                "question": (
                    "Are emergent abilities in large models real or "
                    "artifacts of measurement?"
                ),
                "expected_claim": (
                    "P003 claims emergent abilities exist at scale "
                    "thresholds, but P004 shows they disappear with "
                    "continuous metrics."
                ),
                "required_papers": ["P003", "P004"],
                "keywords": [
                    "emergent",
                    "metric",
                    "continuous",
                    "artifact",
                    "contradict",
                ],
            },
            {
                "question": (
                    "What is the best approach to alignment: RLHF, "
                    "DPO, or Constitutional AI?"
                ),
                "expected_claim": (
                    "RLHF (P005) works but DPO (P009) achieves "
                    "similar results with less compute and no reward "
                    "model. CAI (P008) reduces human annotation needs."
                ),
                "required_papers": ["P005", "P008", "P009"],
                "keywords": [
                    "rlhf",
                    "dpo",
                    "reward model",
                    "constitutional",
                    "compute",
                ],
            },
            {
                "question": (
                    "Can small models match large models through "
                    "augmentation or distillation?"
                ),
                "expected_claim": (
                    "Yes: distillation (P007), RAG (P010), tool-use "
                    "(P011), and test-time compute (P016) all enable "
                    "small models to match large. But P012 argues "
                    "scaling will eventually subsume these approaches."
                ),
                "required_papers": ["P007", "P010", "P011", "P016"],
                "keywords": [
                    "distillation",
                    "rag",
                    "tool",
                    "test-time",
                    "augmentation",
                ],
            },
            {
                "question": (
                    "What are the practical strategies for deploying "
                    "capable models efficiently?"
                ),
                "expected_claim": (
                    "LoRA fine-tuning (P013) with model merging (P018) "
                    "enables efficient deployment. MoE (P006) provides "
                    "sparse scaling. Alignment tax is minimal (P019)."
                ),
                "required_papers": ["P006", "P013", "P018", "P019"],
                "keywords": [
                    "lora",
                    "merging",
                    "moe",
                    "alignment tax",
                    "efficient",
                ],
            },
        ],
    },
    # Scenario 1: medical research
    {
        "papers": [
            {
                "id": "M001",
                "title": "Vitamin D Supplementation and Immune Function",
                "abstract": (
                    "Meta-analysis of 25 RCTs shows vitamin D "
                    "supplementation reduces respiratory infection "
                    "risk by 12% overall."
                ),
                "full_text": (
                    "Subgroup analysis reveals the 12% reduction is "
                    "driven entirely by participants with baseline "
                    "deficiency (<25 nmol/L). In sufficient participants "
                    "(>50 nmol/L), no benefit was observed. Daily dosing "
                    "was more effective than bolus dosing."
                ),
                "findings": [
                    "12% reduction in respiratory infections",
                    "benefit only in deficient individuals",
                    "daily dosing superior to bolus",
                ],
                "tags": ["vitamin-d", "immune", "respiratory"],
            },
            {
                "id": "M002",
                "title": "High-Dose Vitamin D Shows No Benefit",
                "abstract": (
                    "A large RCT (n=25,000) finds no benefit of "
                    "high-dose monthly vitamin D (100,000 IU) on "
                    "respiratory infections, even in deficient subgroup."
                ),
                "full_text": (
                    "Contradicting M001's meta-analysis, our single "
                    "large RCT with monthly 100,000 IU vitamin D shows "
                    "no reduction in respiratory infections (HR 0.98, "
                    "95% CI 0.92-1.05). However, this used bolus dosing "
                    "which M001 found inferior. The contradiction may "
                    "be explained by dosing frequency."
                ),
                "findings": [
                    "no benefit at high monthly dose",
                    "contradicts M001 but used inferior dosing schedule",
                ],
                "tags": ["vitamin-d", "immune", "rct"],
            },
            {
                "id": "M003",
                "title": "Gut Microbiome and Immune Regulation",
                "abstract": (
                    "Gut microbiome diversity is the strongest predictor "
                    "of immune response to vaccination, stronger than "
                    "age or BMI."
                ),
                "full_text": (
                    "Analysis of 1,000 participants shows microbiome "
                    "Shannon diversity index predicts antibody response "
                    "to flu vaccine (R=0.45, p<0.001). Participants "
                    "with low diversity had 40% lower antibody titers. "
                    "Probiotic supplementation did not improve diversity "
                    "or vaccine response."
                ),
                "findings": [
                    "microbiome diversity predicts vaccine response",
                    "probiotics did not help",
                ],
                "tags": ["microbiome", "immune", "vaccination"],
            },
            {
                "id": "M004",
                "title": "Probiotic Supplementation Improves Vaccine Response",
                "abstract": (
                    "A 12-week probiotic course before vaccination "
                    "improved antibody response by 25% in elderly "
                    "patients."
                ),
                "full_text": (
                    "Contradicting M003's finding that probiotics did "
                    "not help, our study in elderly (>65) shows "
                    "Lactobacillus rhamnosus supplementation for 12 "
                    "weeks improved flu vaccine antibody titers by 25% "
                    "(p=0.01). The difference may be age-related: M003 "
                    "studied younger adults (25-55)."
                ),
                "findings": [
                    "probiotics help elderly vaccine response",
                    "contradicts M003 but different age group",
                ],
                "tags": ["probiotics", "immune", "elderly"],
            },
            {
                "id": "M005",
                "title": "Exercise and Immune Function: Dose-Response",
                "abstract": (
                    "Moderate exercise (150 min/week) enhances immune "
                    "function, but high-intensity training (>300 min/"
                    "week) temporarily suppresses it."
                ),
                "full_text": (
                    "Study of 3,000 adults shows J-shaped relationship: "
                    "moderate exercisers had 30% fewer sick days than "
                    "sedentary. Athletes training >300 min/week had "
                    "similar sick days as sedentary. The 'open window' "
                    "of immunosuppression lasts 3-72 hours post-exercise."
                ),
                "findings": [
                    "J-shaped exercise-immune relationship",
                    "moderate exercise reduces illness 30%",
                    "heavy exercise increases vulnerability",
                ],
                "tags": ["exercise", "immune", "dose-response"],
            },
            {
                "id": "M006",
                "title": "No Open Window: Exercise Immunosuppression Myth",
                "abstract": (
                    "The post-exercise immunosuppression 'open window' "
                    "theory is not supported by modern evidence. "
                    "Exercise is beneficial at all intensities."
                ),
                "full_text": (
                    "Re-analysis of M005's data plus 15 additional "
                    "studies shows the J-curve disappears when "
                    "controlling for sleep, stress, and travel "
                    "(confounders in athlete populations). Exercise "
                    "at all intensities enhances immune function. "
                    "This directly contradicts M005."
                ),
                "findings": [
                    "no immunosuppression from heavy exercise",
                    "J-curve explained by confounders",
                    "contradicts M005",
                ],
                "tags": ["exercise", "immune", "methodology"],
            },
            {
                "id": "M007",
                "title": "Sleep Duration and Infection Risk",
                "abstract": (
                    "Sleeping <6 hours/night increases susceptibility "
                    "to the common cold by 4.2x compared to >7 hours."
                ),
                "full_text": (
                    "Prospective study (n=164) with objective sleep "
                    "measurement and rhinovirus challenge. Participants "
                    "sleeping <6h had 4.2x higher infection rate than "
                    ">7h sleepers (p<0.001). Sleep efficiency (<92%) "
                    "was also a significant predictor (2.5x risk)."
                ),
                "findings": [
                    "short sleep 4.2x cold risk",
                    "sleep efficiency also matters",
                ],
                "tags": ["sleep", "immune", "infection"],
            },
            {
                "id": "M008",
                "title": "Zinc Supplementation for Cold Prevention",
                "abstract": (
                    "Zinc lozenges started within 24h of cold onset "
                    "reduce duration by 33%. Preventive daily zinc "
                    "reduces incidence by 28%."
                ),
                "full_text": (
                    "Meta-analysis of 15 RCTs. Therapeutic zinc (>75mg/"
                    "day as lozenges) reduces cold duration from 7 to "
                    "4.7 days. Preventive zinc (15-30mg/day) reduces "
                    "annual cold incidence from 2.1 to 1.5 episodes. "
                    "Zinc acetate more effective than zinc gluconate."
                ),
                "findings": [
                    "zinc reduces cold duration 33%",
                    "preventive zinc reduces incidence 28%",
                    "zinc acetate superior to gluconate",
                ],
                "tags": ["zinc", "cold", "supplement"],
            },
            {
                "id": "M009",
                "title": "Zinc Supplementation: Updated Meta-Analysis",
                "abstract": (
                    "Updating prior meta-analyses with 8 new RCTs, "
                    "zinc benefit shrinks to 15% duration reduction "
                    "and is not statistically significant."
                ),
                "full_text": (
                    "Including 8 RCTs published after M008's analysis, "
                    "the pooled effect shrinks from 33% to 15% (95% CI "
                    "-2% to 30%, p=0.08). Publication bias likely "
                    "inflated M008's estimate. When restricting to "
                    "low-risk-of-bias studies, the effect is 10% and "
                    "non-significant."
                ),
                "findings": [
                    "zinc benefit smaller than M008 claimed",
                    "publication bias inflated earlier estimate",
                    "effect non-significant in high-quality studies",
                ],
                "tags": ["zinc", "cold", "meta-analysis"],
            },
            {
                "id": "M010",
                "title": "Stress and Immune Biomarkers",
                "abstract": (
                    "Chronic psychological stress reduces NK cell "
                    "activity by 15-25% and increases inflammatory "
                    "markers (IL-6, CRP)."
                ),
                "full_text": (
                    "Longitudinal study of 300 caregivers vs controls "
                    "over 3 years. Chronic stress reduced NK cell "
                    "cytotoxicity by 23% (p<0.001) and doubled IL-6 "
                    "levels. Acute stress had opposite effect (brief "
                    "immune enhancement). Mindfulness reduced stress "
                    "biomarkers by 20%."
                ),
                "findings": [
                    "chronic stress reduces NK cell activity 23%",
                    "acute stress briefly enhances immunity",
                    "mindfulness reduces stress biomarkers 20%",
                ],
                "tags": ["stress", "immune", "inflammation"],
            },
            {
                "id": "M011",
                "title": "Cold Exposure and Immune Function",
                "abstract": (
                    "Regular cold exposure (cold showers, winter "
                    "swimming) reduces sick days by 29% in a large "
                    "Dutch RCT."
                ),
                "full_text": (
                    "RCT (n=3,018) assigned participants to 30/60/90 "
                    "second cold showers for 30 days. All groups had "
                    "29% fewer sick days (p<0.01) vs control. Duration "
                    "didn't matter. Mechanism unclear but may involve "
                    "norepinephrine-mediated immune activation."
                ),
                "findings": [
                    "cold showers reduce sick days 29%",
                    "duration of cold exposure doesn't matter",
                ],
                "tags": ["cold-exposure", "immune", "intervention"],
            },
            {
                "id": "M012",
                "title": "Immunosenescence and Aging",
                "abstract": (
                    "Immune function declines significantly after age "
                    "60, with T-cell diversity dropping by 50% per "
                    "decade."
                ),
                "full_text": (
                    "Cross-sectional study of 5,000 adults aged 20-90. "
                    "T-cell receptor diversity drops 50% per decade "
                    "after 60. Vaccine efficacy drops from 70% (young) "
                    "to 30% (elderly). This contextualizes M004's "
                    "finding that elderly benefit most from immune "
                    "interventions (probiotics, supplements)."
                ),
                "findings": [
                    "T-cell diversity drops 50% per decade after 60",
                    "vaccine efficacy drops significantly with age",
                ],
                "tags": ["aging", "immune", "T-cells"],
            },
            {
                "id": "M013",
                "title": "Mediterranean Diet and Immune Markers",
                "abstract": (
                    "Adherence to Mediterranean diet improves immune "
                    "markers and reduces CRP by 20% over 12 months."
                ),
                "full_text": (
                    "RCT (n=600) comparing Mediterranean vs Western "
                    "diet for 12 months. Mediterranean group showed "
                    "20% CRP reduction, 15% NK cell activity increase, "
                    "and 30% higher microbiome diversity. Supports "
                    "M003's finding that microbiome diversity predicts "
                    "immune function."
                ),
                "findings": [
                    "Mediterranean diet reduces CRP 20%",
                    "increases microbiome diversity 30%",
                    "supports microbiome-immune link",
                ],
                "tags": ["diet", "immune", "microbiome"],
            },
            {
                "id": "M014",
                "title": "Vitamin C Meta-Analysis: Modest Effects",
                "abstract": (
                    "Vitamin C reduces cold duration by 8% in adults "
                    "and 14% in children. No effect on incidence "
                    "except in extreme physical stress."
                ),
                "full_text": (
                    "Cochrane review of 29 RCTs (11,306 participants). "
                    "Regular vitamin C (>200mg/day) reduces cold duration "
                    "8% in adults, 14% in children. No effect on "
                    "incidence in general population. Significant "
                    "incidence reduction (50%) only in marathon runners "
                    "and soldiers under extreme physical stress."
                ),
                "findings": [
                    "vitamin C modestly reduces cold duration",
                    "no incidence effect except extreme exercise",
                ],
                "tags": ["vitamin-c", "cold", "supplement"],
            },
            {
                "id": "M015",
                "title": "Intermittent Fasting and Immune Regeneration",
                "abstract": (
                    "72-hour fasting cycles trigger stem cell-based "
                    "immune system regeneration in mice and early "
                    "human trials."
                ),
                "full_text": (
                    "Prolonged fasting reduces PKA signaling, "
                    "triggering hematopoietic stem cell regeneration "
                    "of immune cells. In mice, 48-72h fasting cycles "
                    "reversed immunosuppression from chemotherapy. "
                    "Phase I human trial (n=30) showed 25% increase in "
                    "white blood cell count after 3 fasting cycles. "
                    "Safety and efficacy in larger trials needed."
                ),
                "findings": [
                    "prolonged fasting triggers immune regeneration",
                    "preliminary human evidence promising",
                ],
                "tags": ["fasting", "immune", "regeneration"],
            },
            {
                "id": "M016",
                "title": "Intermittent Fasting: No Immune Benefit in Humans",
                "abstract": (
                    "Large RCT (n=500) finds no immune benefit of "
                    "16:8 intermittent fasting over 6 months."
                ),
                "full_text": (
                    "Contradicting M015's promising early results, our "
                    "large RCT of 16:8 IF found no changes in WBC count, "
                    "NK cell activity, or inflammatory markers vs "
                    "controls over 6 months. However, this used 16:8 "
                    "IF (daily), not the 72-hour prolonged fasting "
                    "protocol that M015 tested. The protocols are "
                    "fundamentally different."
                ),
                "findings": [
                    "16:8 IF shows no immune benefit",
                    "different protocol than M015's prolonged fasting",
                ],
                "tags": ["fasting", "immune", "rct"],
            },
            {
                "id": "M017",
                "title": "Personalized Immune Optimization",
                "abstract": (
                    "Machine learning models integrating sleep, stress, "
                    "diet, and microbiome data predict individual immune "
                    "response with 80% accuracy."
                ),
                "full_text": (
                    "Combining data from M003, M007, M010, and M013, "
                    "a random forest model predicts vaccine response "
                    "with AUC=0.82. Top features: microbiome diversity, "
                    "sleep duration, chronic stress score, and "
                    "Mediterranean diet adherence. This suggests "
                    "personalized protocols may be more effective than "
                    "universal supplement recommendations."
                ),
                "findings": [
                    "ML can predict immune response 80% accuracy",
                    "personalized approach may outperform universal supplements",
                ],
                "tags": ["personalized", "ml", "immune"],
            },
            {
                "id": "M018",
                "title": "Elderberry Extract for Cold and Flu",
                "abstract": (
                    "Elderberry extract reduces cold/flu severity and "
                    "duration by 2 days in a meta-analysis of 5 RCTs."
                ),
                "full_text": (
                    "Meta-analysis of 5 RCTs (n=540) shows elderberry "
                    "supplementation reduces upper respiratory symptom "
                    "duration by 2 days (95% CI 1.3-2.7). Effect size "
                    "larger for flu (3 days) than cold (1.5 days). "
                    "All studies were small and 3 were industry-funded."
                ),
                "findings": [
                    "elderberry reduces cold/flu duration by 2 days",
                    "stronger effect for flu",
                    "evidence quality concerns (small, industry-funded)",
                ],
                "tags": ["elderberry", "cold", "supplement"],
            },
            {
                "id": "M019",
                "title": "Sauna Bathing and Immune Function",
                "abstract": (
                    "Regular sauna use (4-7x/week) reduces pneumonia "
                    "risk by 37% and increases white blood cell count."
                ),
                "full_text": (
                    "Finnish cohort study (n=2,210, 25-year follow-up). "
                    "Sauna frequency inversely associated with pneumonia "
                    "risk: 4-7x/week had 37% lower risk vs 1x/week "
                    "(HR 0.63, 95% CI 0.45-0.88). Mechanism: heat "
                    "exposure stimulates WBC production and HSP release. "
                    "Similar mechanism proposed for M011's cold exposure."
                ),
                "findings": [
                    "sauna 4-7x/week reduces pneumonia risk 37%",
                    "mechanism via WBC stimulation",
                ],
                "tags": ["sauna", "immune", "heat-exposure"],
            },
            {
                "id": "M020",
                "title": "Social Connection and Immune Function",
                "abstract": (
                    "Loneliness is associated with 26% higher "
                    "inflammatory markers and reduced antiviral "
                    "gene expression."
                ),
                "full_text": (
                    "Study of 200 adults shows lonely individuals have "
                    "26% higher CRP, 20% higher IL-6, and conserved "
                    "transcriptional response to adversity (CTRA) "
                    "gene expression pattern. This pattern upregulates "
                    "inflammation genes and downregulates antiviral "
                    "genes. Relates to M010's stress findings -- "
                    "loneliness may be a chronic stressor."
                ),
                "findings": [
                    "loneliness increases inflammation 26%",
                    "reduces antiviral gene expression",
                    "loneliness as chronic stressor",
                ],
                "tags": ["social", "immune", "inflammation"],
            },
        ],
        "questions": [
            {
                "question": (
                    "Does vitamin D supplementation prevent "
                    "respiratory infections?"
                ),
                "expected_claim": (
                    "M001 shows 12% reduction overall but only in "
                    "deficient individuals with daily dosing. M002 "
                    "found no benefit with monthly bolus dosing."
                ),
                "required_papers": ["M001", "M002"],
                "keywords": [
                    "deficien",
                    "daily",
                    "bolus",
                    "contradict",
                    "dosing",
                ],
            },
            {
                "question": (
                    "Does exercise intensity affect immune function, "
                    "and is there an 'open window' of vulnerability?"
                ),
                "expected_claim": (
                    "M005 claimed a J-shaped relationship with heavy "
                    "exercise suppressing immunity, but M006 showed "
                    "this was due to confounders."
                ),
                "required_papers": ["M005", "M006"],
                "keywords": [
                    "j-shaped",
                    "open window",
                    "confounder",
                    "contradict",
                ],
            },
            {
                "question": (
                    "What is the role of the gut microbiome in "
                    "immune function?"
                ),
                "expected_claim": (
                    "Microbiome diversity predicts immune response "
                    "(M003), Mediterranean diet improves diversity "
                    "(M013). Probiotics help elderly (M004) but not "
                    "younger adults (M003)."
                ),
                "required_papers": ["M003", "M004", "M013"],
                "keywords": [
                    "diversity",
                    "microbiome",
                    "probiotic",
                    "elderly",
                    "diet",
                ],
            },
            {
                "question": (
                    "What supplements have the strongest evidence "
                    "for immune support?"
                ),
                "expected_claim": (
                    "Zinc (M008) had strong evidence but updated "
                    "meta-analysis (M009) shrank the effect. Vitamin "
                    "C (M014) is modest. Elderberry (M018) promising "
                    "but low-quality evidence."
                ),
                "required_papers": ["M008", "M009", "M014"],
                "keywords": [
                    "zinc",
                    "vitamin c",
                    "elderberry",
                    "publication bias",
                    "modest",
                ],
            },
            {
                "question": (
                    "What lifestyle factors most effectively boost "
                    "immune function?"
                ),
                "expected_claim": (
                    "Sleep (M007, 4.2x risk), stress management "
                    "(M010), exercise (M005/M006 once resolved), diet "
                    "(M013), and social connection (M020). ML model "
                    "(M017) suggests personalized approach."
                ),
                "required_papers": ["M007", "M010", "M013", "M017"],
                "keywords": [
                    "sleep",
                    "stress",
                    "exercise",
                    "diet",
                    "personalized",
                ],
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Task implementation
# ---------------------------------------------------------------------------
@dataclass
class ResearchAssistantTask:
    """Research assistant benchmark with iterative refinement."""

    seed: int = 0
    _step: int = 0
    _scenario: dict = field(default_factory=dict)
    _papers: list[dict] = field(default_factory=list)
    _questions: list[dict] = field(default_factory=list)
    _current_question_idx: int = 0
    _findings: dict[str, dict] = field(default_factory=dict)
    _next_finding_id: int = 1
    _papers_read: set[str] = field(default_factory=set)
    _abstracts_read: set[str] = field(default_factory=set)
    _done: bool = False
    _max_steps: int = 100
    _tool_errors: int = 0
    _revision_count: int = 0
    _rng: random.Random = field(default_factory=random.Random)

    TOOLS: list[dict] = field(
        default_factory=lambda: [
            {
                "type": "function",
                "function": {
                    "name": "search_papers",
                    "description": (
                        "Search the knowledge base for papers by query."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_abstract",
                    "description": "Read a paper's abstract by ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "paper_id": {
                                "type": "string",
                                "description": "Paper ID (e.g. P001)",
                            },
                        },
                        "required": ["paper_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_full",
                    "description": "Read a paper's full text by ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "paper_id": {
                                "type": "string",
                                "description": "Paper ID (e.g. P001)",
                            },
                        },
                        "required": ["paper_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "submit_finding",
                    "description": (
                        "Submit a research finding with evidence."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "claim": {
                                "type": "string",
                                "description": "The research finding",
                            },
                            "evidence": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Paper IDs supporting claim",
                            },
                        },
                        "required": ["claim", "evidence"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "revise_finding",
                    "description": (
                        "Revise a previously submitted finding."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "finding_id": {
                                "type": "string",
                                "description": "Finding ID to revise",
                            },
                            "new_claim": {
                                "type": "string",
                                "description": "Updated claim",
                            },
                            "new_evidence": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Updated paper IDs",
                            },
                        },
                        "required": [
                            "finding_id",
                            "new_claim",
                            "new_evidence",
                        ],
                    },
                },
            },
        ]
    )

    def setup(self) -> str:
        self._rng = random.Random(self.seed)
        idx = self.seed % len(_SCENARIOS)
        self._scenario = _SCENARIOS[idx]
        self._papers = self._scenario["papers"]
        self._questions = self._scenario["questions"]
        self._current_question_idx = 0
        self._findings = {}
        self._next_finding_id = 1
        self._papers_read = set()
        self._abstracts_read = set()
        self._step = 0
        self._done = False
        self._tool_errors = 0
        self._revision_count = 0

        paper_list = "\n".join(
            f"  [{p['id']}] {p['title']}" for p in self._papers
        )
        first_q = self._questions[0]["question"]

        return (
            "You are a research assistant. Your knowledge base "
            f"contains {len(self._papers)} papers:\n\n"
            f"{paper_list}\n\n"
            "You will answer 5 research questions sequentially. "
            "Use search_papers, read_abstract, and read_full to "
            "investigate, then submit_finding with your claim and "
            "evidence (paper IDs). If later evidence contradicts an "
            "earlier finding, use revise_finding.\n\n"
            f"Question 1: {first_q}"
        )

    def execute_action(self, action: str) -> str:
        self._step += 1
        if self._step >= self._max_steps:
            self._done = True
            return "Maximum steps reached."

        tool_name, args = _parse_tool_call(action)

        if tool_name == "search_papers":
            return self._handle_search(args)
        if tool_name == "read_abstract":
            return self._handle_read_abstract(args)
        if tool_name == "read_full":
            return self._handle_read_full(args)
        if tool_name == "submit_finding":
            return self._handle_submit(args)
        if tool_name == "revise_finding":
            return self._handle_revise(args)

        self._tool_errors += 1
        return (
            f"Error: Unknown tool '{tool_name}'. Available: "
            "search_papers, read_abstract, read_full, "
            "submit_finding, revise_finding"
        )

    def _handle_search(self, args: dict) -> str:
        query = args.get("query", "").lower()
        matches: list[dict[str, str]] = []
        query_words = query.split()
        for p in self._papers:
            searchable = (
                f"{p['title']} {p['abstract']} "
                f"{' '.join(p['tags'])}"
            ).lower()
            if any(w in searchable for w in query_words):
                matches.append({
                    "id": p["id"],
                    "title": p["title"],
                    "tags": p["tags"],
                })
        if not matches:
            return json.dumps({"results": [], "total": 0})
        return json.dumps(
            {"results": matches[:10], "total": len(matches)}
        )

    def _handle_read_abstract(self, args: dict) -> str:
        paper_id = args.get("paper_id", "")
        for p in self._papers:
            if p["id"] == paper_id:
                self._abstracts_read.add(paper_id)
                return json.dumps({
                    "id": p["id"],
                    "title": p["title"],
                    "abstract": p["abstract"],
                })
        return json.dumps(
            {"error": f"Paper '{paper_id}' not found."}
        )

    def _handle_read_full(self, args: dict) -> str:
        paper_id = args.get("paper_id", "")
        for p in self._papers:
            if p["id"] == paper_id:
                self._papers_read.add(paper_id)
                self._abstracts_read.add(paper_id)
                return json.dumps({
                    "id": p["id"],
                    "title": p["title"],
                    "abstract": p["abstract"],
                    "full_text": p["full_text"],
                    "findings": p["findings"],
                })
        return json.dumps(
            {"error": f"Paper '{paper_id}' not found."}
        )

    def _handle_submit(self, args: dict) -> str:
        claim = args.get("claim", "")
        evidence = args.get("evidence", [])
        finding_id = f"F{self._next_finding_id}"
        self._next_finding_id += 1
        self._findings[finding_id] = {
            "claim": claim,
            "evidence": evidence,
            "question_idx": self._current_question_idx,
            "revised": False,
        }

        # Advance to next question
        self._current_question_idx += 1
        if self._current_question_idx >= len(self._questions):
            self._done = True
            return (
                f"Finding {finding_id} recorded. "
                "All questions answered. Research complete."
            )

        next_q = self._questions[self._current_question_idx]
        return (
            f"Finding {finding_id} recorded.\n\n"
            f"Question {self._current_question_idx + 1}: "
            f"{next_q['question']}"
        )

    def _handle_revise(self, args: dict) -> str:
        finding_id = args.get("finding_id", "")
        new_claim = args.get("new_claim", "")
        new_evidence = args.get("new_evidence", [])

        if finding_id not in self._findings:
            return f"Error: Finding '{finding_id}' not found."

        self._findings[finding_id]["claim"] = new_claim
        self._findings[finding_id]["evidence"] = new_evidence
        self._findings[finding_id]["revised"] = True
        self._revision_count += 1

        return f"Finding {finding_id} revised."

    def is_complete(self) -> bool:
        return self._done or self._step >= self._max_steps

    def score(self) -> TaskResult:
        total_questions = len(self._questions)
        findings_by_q: dict[int, dict] = {}
        for fid, f in self._findings.items():
            qi = f["question_idx"]
            # Take the latest finding per question
            findings_by_q[qi] = {"id": fid, **f}

        correct_findings = 0
        evidence_score_total = 0.0

        for qi, q in enumerate(self._questions):
            if qi not in findings_by_q:
                continue
            finding = findings_by_q[qi]

            # Check claim quality via keywords
            claim_lower = finding["claim"].lower()
            keywords = q["keywords"]
            matched_kw = sum(
                1 for kw in keywords if kw.lower() in claim_lower
            )
            kw_ratio = matched_kw / len(keywords) if keywords else 0.0
            if kw_ratio >= 0.4:  # at least 40% of keywords present
                correct_findings += 1

            # Evidence quality: cited the required papers?
            required = set(q["required_papers"])
            cited = set(finding["evidence"])
            if required:
                evidence_score_total += (
                    len(required & cited) / len(required)
                )

        findings_accuracy = (
            correct_findings / total_questions * 100
            if total_questions > 0
            else 0.0
        )
        evidence_accuracy = (
            evidence_score_total / total_questions * 100
            if total_questions > 0
            else 0.0
        )
        # Combined: 50% findings + 50% evidence
        accuracy = findings_accuracy * 0.5 + evidence_accuracy * 0.5

        return TaskResult(
            completion=(
                len(findings_by_q) == total_questions
                and correct_findings >= total_questions // 2
            ),
            accuracy=accuracy,
            steps=self._step,
            tool_errors=self._tool_errors,
            extra={
                "findings_correct": correct_findings,
                "total_questions": total_questions,
                "findings_submitted": len(self._findings),
                "revisions": self._revision_count,
                "papers_read_full": len(self._papers_read),
                "abstracts_read": len(self._abstracts_read),
                "evidence_accuracy": evidence_accuracy,
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_tool_call(action: str) -> tuple[str, dict]:
    """Parse a tool call from the agent's response."""
    try:
        parsed = json.loads(action)
        if isinstance(parsed, dict):
            name = parsed.get("tool") or parsed.get("name", "")
            args = parsed.get("arguments") or parsed.get("args", {})
            if isinstance(args, str):
                args = json.loads(args)
            return name, args
    except (json.JSONDecodeError, TypeError):
        pass
    return "", {}
