# Phase 44: Remaining Domain Schemas + Developer Documentation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Ship the remaining five built-in domains (code, research,
collab, customer, creative), full developer documentation for the
schema extension API, cookbook recipes, and README updates.

**Depends on:** Phase 43 (built-in schemas + context packer).

**Architecture:**
- `src/soma/schemas/builtin/code.py` — Decision, Pattern, Incident, DependencyNote
- `src/soma/schemas/builtin/research.py` — Hypothesis, Experiment, Result, Literature
- `src/soma/schemas/builtin/collab.py` — ActionItem, Decision, FollowUp, StakeholderPosition
- `src/soma/schemas/builtin/customer.py` — Profile, Issue, Sentiment, Preference
- `src/soma/schemas/builtin/creative.py` — Character, WorldDetail, Continuity, PlotThread
- `docs/schemas.md` — full developer guide (how to define, store, retrieve, extend, pack context)
- `docs/cookbook.md` — recipes §24-26 (agent workflow + custom schema + context packing)
- `README.md` — feature table update + schema-framework mention

---

### Tasks (4):
1. Code + research domain schemas + tests
2. Collab + customer + creative domain schemas + tests
3. Full developer documentation (`docs/schemas.md`)
4. Cookbook recipes + README cross-refs

Each task = one commit. ~1 week total.
