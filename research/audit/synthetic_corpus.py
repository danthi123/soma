"""Shared synthetic corpus and recall measurement for audit phases 2-5.

Provides:
- PHASE1_FACTS: 50 diverse facts (preferences, events, people, places)
- FLOOD_FACTS: 200 interference facts on different topics
- QUERIES: 20 (query, expected_substring) pairs targeting phase-1 facts
- measure_recall(): checks if the correct answer appears in top-k retrieval
- build_memory_layer(): convenience constructor with sbert embeddings
- build_soma_stack(): convenience constructor for SOMA + tokenizer + encoder
"""

from __future__ import annotations

import time
from typing import Any

import torch

from soma.core.config import SOMAConfig
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.memory.api import MemoryLayer
from soma.system import SOMA

# ---------------------------------------------------------------------------
# Phase-1 facts: 50 diverse memories
# ---------------------------------------------------------------------------
PHASE1_FACTS: list[str] = [
    # Preferences (10)
    "User's favorite color is blue.",
    "User prefers dark mode in all applications.",
    "User's favorite cuisine is Japanese, especially ramen.",
    "User drinks two cups of black coffee every morning.",
    "User's favorite programming language is Python.",
    "User prefers reading physical books over ebooks.",
    "User's favorite movie is Blade Runner 2049.",
    "User listens to jazz while working.",
    "User prefers window seats on flights.",
    "User's favorite season is autumn.",
    # Events (10)
    "Meeting with Bob scheduled for Tuesday at 3pm.",
    "User has a dentist appointment on March 15th.",
    "The team offsite is planned for the second week of April.",
    "User's birthday is on September 22nd.",
    "Project deadline for Phoenix is June 30th.",
    "User signed up for a marathon on November 5th.",
    "User's anniversary is on Valentine's Day, February 14th.",
    "Conference talk submission due by May 1st.",
    "User booked a flight to Tokyo for July 10th.",
    "User's car insurance renewal is in August.",
    # People (10)
    "Alice is the user's project manager at work.",
    "Bob is the user's closest friend from college.",
    "Dr. Sarah Chen is the user's physician.",
    "Carlos manages the backend team at the user's company.",
    "Emma is the user's sister who lives in Portland.",
    "Frank is the user's personal trainer at the gym.",
    "Grace is the user's neighbor who has a golden retriever.",
    "Henry is the user's accountant for tax season.",
    "Irene is the user's mentor from the previous job.",
    "Jake is the user's roommate who works night shifts.",
    # Places (10)
    "User lives in a two-bedroom apartment in downtown Seattle.",
    "User works at the TechCorp office on 5th Avenue.",
    "User's favorite coffee shop is Mocha House on Pine Street.",
    "User's gym is CrossFit Central, three blocks from home.",
    "User's parents live in a farmhouse outside Austin, Texas.",
    "User frequents Green Park for weekend jogging.",
    "User's preferred grocery store is Whole Foods on Broadway.",
    "User's dentist office is at Smile Clinic on Oak Avenue.",
    "User vacations at a cabin near Lake Tahoe every winter.",
    "User's favorite restaurant is Sakura Sushi on Main Street.",
    # Technical / work (10)
    "User's current project uses a microservices architecture.",
    "The production database is PostgreSQL 15 on AWS RDS.",
    "User's team follows two-week sprint cycles.",
    "The CI/CD pipeline uses GitHub Actions with Docker.",
    "User's laptop is a MacBook Pro M2 with 32GB RAM.",
    "The staging environment URL is staging.techcorp.internal.",
    "User uses VS Code with Vim keybindings.",
    "The team's Slack channel for alerts is #ops-alerts.",
    "User's SSH key fingerprint starts with SHA256:abc123.",
    "The monitoring stack is Prometheus plus Grafana.",
]

# ---------------------------------------------------------------------------
# Flood facts: 200 interference memories on unrelated topics
# ---------------------------------------------------------------------------
_FLOOD_TEMPLATES: list[str] = [
    "The capital of {country} is {capital}.",
    "The chemical symbol for {element} is {symbol}.",
    "{animal} can live up to {years} years in the wild.",
    "The {sport} world record is held by {athlete}.",
    "The {river} river flows through {city}.",
    "Mount {mountain} has an elevation of {height} meters.",
    "{composer} composed the famous piece {piece}.",
    "The {language} language has approximately {speakers} million speakers.",
    "{planet} has {moons} known moons.",
    "The {building} was completed in {year}.",
]

_FLOOD_DATA: list[dict[str, str]] = [
    {"country": "Mongolia", "capital": "Ulaanbaatar"},
    {"country": "Bhutan", "capital": "Thimphu"},
    {"country": "Suriname", "capital": "Paramaribo"},
    {"country": "Lesotho", "capital": "Maseru"},
    {"country": "Tuvalu", "capital": "Funafuti"},
    {"country": "Liechtenstein", "capital": "Vaduz"},
    {"country": "Comoros", "capital": "Moroni"},
    {"country": "Djibouti", "capital": "Djibouti City"},
    {"country": "Eritrea", "capital": "Asmara"},
    {"country": "Kiribati", "capital": "Tarawa"},
    {"country": "Nauru", "capital": "Yaren"},
    {"country": "Palau", "capital": "Ngerulmud"},
    {"country": "Tonga", "capital": "Nukualofa"},
    {"country": "Vanuatu", "capital": "Port Vila"},
    {"country": "Samoa", "capital": "Apia"},
    {"country": "Brunei", "capital": "Bandar Seri Begawan"},
    {"country": "Andorra", "capital": "Andorra la Vella"},
    {"country": "San Marino", "capital": "San Marino City"},
    {"country": "Seychelles", "capital": "Victoria"},
    {"country": "Maldives", "capital": "Male"},
    {"element": "Ruthenium", "symbol": "Ru"},
    {"element": "Rhodium", "symbol": "Rh"},
    {"element": "Palladium", "symbol": "Pd"},
    {"element": "Osmium", "symbol": "Os"},
    {"element": "Iridium", "symbol": "Ir"},
    {"element": "Hafnium", "symbol": "Hf"},
    {"element": "Tantalum", "symbol": "Ta"},
    {"element": "Rhenium", "symbol": "Re"},
    {"element": "Thallium", "symbol": "Tl"},
    {"element": "Bismuth", "symbol": "Bi"},
    {"element": "Polonium", "symbol": "Po"},
    {"element": "Francium", "symbol": "Fr"},
    {"element": "Radium", "symbol": "Ra"},
    {"element": "Actinium", "symbol": "Ac"},
    {"element": "Protactinium", "symbol": "Pa"},
    {"element": "Neptunium", "symbol": "Np"},
    {"element": "Americium", "symbol": "Am"},
    {"element": "Curium", "symbol": "Cm"},
    {"element": "Berkelium", "symbol": "Bk"},
    {"element": "Californium", "symbol": "Cf"},
    {"animal": "Galapagos tortoise", "years": "175"},
    {"animal": "Bowhead whale", "years": "200"},
    {"animal": "Greenland shark", "years": "400"},
    {"animal": "Koi fish", "years": "200"},
    {"animal": "Red sea urchin", "years": "200"},
    {"animal": "Macaw", "years": "80"},
    {"animal": "Elephant", "years": "70"},
    {"animal": "Albatross", "years": "60"},
    {"animal": "Crocodile", "years": "70"},
    {"animal": "Sturgeon", "years": "100"},
    {"animal": "Tuatara", "years": "100"},
    {"animal": "Flamingo", "years": "50"},
    {"animal": "Orangutan", "years": "45"},
    {"animal": "Horse", "years": "30"},
    {"animal": "Gorilla", "years": "40"},
    {"animal": "Condor", "years": "75"},
    {"animal": "Swan", "years": "25"},
    {"animal": "Eagle", "years": "30"},
    {"animal": "Parrot", "years": "80"},
    {"animal": "Lobster", "years": "50"},
    {"sport": "100m sprint", "athlete": "Usain Bolt"},
    {"sport": "marathon", "athlete": "Eliud Kipchoge"},
    {"sport": "long jump", "athlete": "Mike Powell"},
    {"sport": "pole vault", "athlete": "Armand Duplantis"},
    {"sport": "shot put", "athlete": "Ryan Crouser"},
    {"sport": "javelin throw", "athlete": "Jan Zelezny"},
    {"sport": "high jump", "athlete": "Javier Sotomayor"},
    {"sport": "discus throw", "athlete": "Jurgen Schult"},
    {"sport": "triple jump", "athlete": "Jonathan Edwards"},
    {"sport": "hammer throw", "athlete": "Yuriy Sedykh"},
    {"sport": "400m sprint", "athlete": "Wayde van Niekerk"},
    {"sport": "800m run", "athlete": "David Rudisha"},
    {"sport": "1500m run", "athlete": "Hicham El Guerrouj"},
    {"sport": "5000m run", "athlete": "Joshua Cheptegei"},
    {"sport": "110m hurdles", "athlete": "Aries Merritt"},
    {"sport": "400m hurdles", "athlete": "Karsten Warholm"},
    {"sport": "decathlon", "athlete": "Kevin Mayer"},
    {"sport": "50km race walk", "athlete": "Yohann Diniz"},
    {"sport": "10000m run", "athlete": "Joshua Cheptegei"},
    {"sport": "steeplechase", "athlete": "Saif Saaeed Shaheen"},
    {"river": "Mekong", "city": "Phnom Penh"},
    {"river": "Danube", "city": "Budapest"},
    {"river": "Volga", "city": "Kazan"},
    {"river": "Zambezi", "city": "Livingstone"},
    {"river": "Indus", "city": "Karachi"},
    {"river": "Yangtze", "city": "Wuhan"},
    {"river": "Congo", "city": "Kinshasa"},
    {"river": "Niger", "city": "Bamako"},
    {"river": "Euphrates", "city": "Baghdad"},
    {"river": "Tigris", "city": "Mosul"},
    {"river": "Rhine", "city": "Cologne"},
    {"river": "Elbe", "city": "Hamburg"},
    {"river": "Seine", "city": "Paris"},
    {"river": "Thames", "city": "London"},
    {"river": "Neva", "city": "Saint Petersburg"},
    {"river": "Tagus", "city": "Lisbon"},
    {"river": "Tiber", "city": "Rome"},
    {"river": "Arno", "city": "Florence"},
    {"river": "Vistula", "city": "Warsaw"},
    {"river": "Douro", "city": "Porto"},
    {"mountain": "Kilimanjaro", "height": "5895"},
    {"mountain": "Elbrus", "height": "5642"},
    {"mountain": "Denali", "height": "6190"},
    {"mountain": "Aconcagua", "height": "6961"},
    {"mountain": "Vinson", "height": "4892"},
    {"mountain": "Puncak Jaya", "height": "4884"},
    {"mountain": "Mont Blanc", "height": "4809"},
    {"mountain": "Matterhorn", "height": "4478"},
    {"mountain": "Fuji", "height": "3776"},
    {"mountain": "Olympus", "height": "2918"},
    {"composer": "Vivaldi", "piece": "The Four Seasons"},
    {"composer": "Debussy", "piece": "Clair de Lune"},
    {"composer": "Chopin", "piece": "Nocturne Op. 9 No. 2"},
    {"composer": "Tchaikovsky", "piece": "Swan Lake"},
    {"composer": "Dvorak", "piece": "New World Symphony"},
    {"composer": "Grieg", "piece": "Peer Gynt Suite"},
    {"composer": "Handel", "piece": "Messiah"},
    {"composer": "Brahms", "piece": "Hungarian Dances"},
    {"composer": "Liszt", "piece": "Hungarian Rhapsody No. 2"},
    {"composer": "Schubert", "piece": "Ave Maria"},
    {"language": "Bengali", "speakers": "230"},
    {"language": "Javanese", "speakers": "82"},
    {"language": "Telugu", "speakers": "83"},
    {"language": "Marathi", "speakers": "83"},
    {"language": "Tamil", "speakers": "78"},
    {"language": "Turkish", "speakers": "80"},
    {"language": "Korean", "speakers": "77"},
    {"language": "Vietnamese", "speakers": "85"},
    {"language": "Italian", "speakers": "67"},
    {"language": "Thai", "speakers": "60"},
    {"planet": "Jupiter", "moons": "95"},
    {"planet": "Saturn", "moons": "146"},
    {"planet": "Uranus", "moons": "28"},
    {"planet": "Neptune", "moons": "16"},
    {"planet": "Mars", "moons": "2"},
    {"planet": "Pluto", "moons": "5"},
    {"planet": "Mercury", "moons": "0"},
    {"planet": "Venus", "moons": "0"},
    {"planet": "Earth", "moons": "1"},
    {"planet": "Eris", "moons": "1"},
    {"building": "Burj Khalifa", "year": "2010"},
    {"building": "Shanghai Tower", "year": "2015"},
    {"building": "Abraj Al-Bait Clock Tower", "year": "2012"},
    {"building": "Taipei 101", "year": "2004"},
    {"building": "One World Trade Center", "year": "2014"},
    {"building": "Lotte World Tower", "year": "2017"},
    {"building": "Petronas Towers", "year": "1998"},
    {"building": "Willis Tower", "year": "1973"},
    {"building": "Empire State Building", "year": "1931"},
    {"building": "CN Tower", "year": "1976"},
]


def _build_flood_facts() -> list[str]:
    """Expand templates x data into 200 flood facts."""
    facts: list[str] = []
    for item in _FLOOD_DATA:
        # Find matching template
        for tmpl in _FLOOD_TEMPLATES:
            # Check if all keys in the item appear in the template
            if all(f"{{{k}}}" in tmpl for k in item):
                facts.append(tmpl.format(**item))
                break
    # Pad to exactly 200 if needed (we have 150 data items, need duplication)
    while len(facts) < 200:
        facts.append(f"Random trivia fact number {len(facts) + 1}: "
                     f"the probability of this being relevant is negligible.")
    return facts[:200]


FLOOD_FACTS: list[str] = _build_flood_facts()

# ---------------------------------------------------------------------------
# Adversarial flood: same topics as phase-1 but WRONG details.
# These compete semantically with the correct memories.
# ---------------------------------------------------------------------------
ADVERSARIAL_FLOOD: list[str] = [
    # Contradicting preferences
    "User's favorite color is red.",
    "User's favorite color is green.",
    "User prefers light mode in all applications.",
    "User's favorite cuisine is Italian, especially pasta.",
    "User's favorite cuisine is Mexican, especially tacos.",
    "User drinks three cups of green tea every morning.",
    "User's favorite programming language is Rust.",
    "User's favorite programming language is JavaScript.",
    "User prefers reading ebooks over physical books.",
    "User's favorite movie is The Matrix.",
    "User's favorite movie is Interstellar.",
    "User listens to classical music while working.",
    "User listens to lo-fi hip hop while working.",
    "User prefers aisle seats on flights.",
    "User's favorite season is spring.",
    "User's favorite season is summer.",
    # Contradicting events
    "Meeting with Bob scheduled for Wednesday at 2pm.",
    "Meeting with Bob scheduled for Thursday at 4pm.",
    "User has a dentist appointment on March 20th.",
    "User has a dentist appointment on April 15th.",
    "The team offsite is planned for the third week of May.",
    "User's birthday is on October 15th.",
    "User's birthday is on September 10th.",
    "Project deadline for Phoenix is July 15th.",
    "Project deadline for Phoenix is May 30th.",
    "User signed up for a marathon on December 1st.",
    "User's anniversary is on March 14th.",
    "Conference talk submission due by June 1st.",
    "User booked a flight to Tokyo for August 5th.",
    "User booked a flight to Seoul for July 10th.",
    "User's car insurance renewal is in September.",
    # Contradicting people
    "Alice is the user's direct report at work.",
    "Bob is the user's colleague from grad school.",
    "Dr. Michael Park is the user's physician.",
    "Carlos manages the frontend team at the user's company.",
    "Emma is the user's sister who lives in Denver.",
    "Frank is the user's personal trainer at home.",
    "Grace is the user's neighbor who has a labrador.",
    "Henry is the user's financial advisor.",
    "Irene is the user's colleague from the current job.",
    "Jake is the user's roommate who works day shifts.",
    # Contradicting places
    "User lives in a studio apartment in downtown Portland.",
    "User lives in a house in suburban Chicago.",
    "User works at the DataCorp office on 3rd Avenue.",
    "User's favorite coffee shop is Bean Counter on Elm Street.",
    "User's gym is Gold's Gym, five blocks from home.",
    "User's parents live in a condo in Miami, Florida.",
    "User frequents Riverside Trail for weekend jogging.",
    "User's preferred grocery store is Trader Joe's on 2nd Ave.",
    "User's dentist office is at Bright Smiles on Maple Drive.",
    "User vacations at a beach house near Malibu every summer.",
    "User's favorite restaurant is Thai Garden on 4th Street.",
    # Contradicting technical
    "User's current project uses a monolithic architecture.",
    "The production database is MySQL 8 on Google Cloud SQL.",
    "User's team follows three-week sprint cycles.",
    "The CI/CD pipeline uses GitLab CI with Kubernetes.",
    "User's laptop is a ThinkPad X1 with 64GB RAM.",
    "The staging environment URL is staging.datacorp.internal.",
    "User uses Neovim with standard keybindings.",
    "The team's Slack channel for alerts is #engineering-alerts.",
    "The main API framework is FastAPI with SQLAlchemy.",
    "User's team uses Jira for task tracking.",
]

# ---------------------------------------------------------------------------
# Query/answer pairs targeting phase-1 facts
# Each answer is a substring that MUST appear in the retrieved text
# ---------------------------------------------------------------------------
QUERIES: list[tuple[str, str]] = [
    ("What is the user's favorite color?", "favorite color is blue"),
    ("What does the user drink in the morning?", "black coffee"),
    ("When is the meeting with Bob?", "Tuesday at 3pm"),
    ("What is the user's favorite movie?", "Blade Runner 2049"),
    ("Who is Alice in the user's life?", "project manager"),
    ("Where does the user live?", "downtown Seattle"),
    ("What database does the production system use?", "PostgreSQL"),
    ("When is the user's birthday?", "September 22nd"),
    ("Who is the user's physician?", "Dr. Sarah Chen"),
    ("What is the user's favorite coffee shop?", "Mocha House"),
    ("What music does the user listen to while working?", "jazz"),
    ("When is the project Phoenix deadline?", "June 30th"),
    ("Who manages the backend team?", "Carlos"),
    ("Where do the user's parents live?", "Austin, Texas"),
    ("What CI/CD system does the team use?", "GitHub Actions"),
    ("What editor does the user use?", "VS Code"),
    ("Where does the user vacation in winter?", "Lake Tahoe"),
    ("Who is the user's roommate?", "Jake"),
    ("What is the user's favorite restaurant?", "Sakura Sushi"),
    ("What is the monitoring stack?", "Prometheus plus Grafana"),
]


def measure_recall(
    mem: MemoryLayer,
    queries: list[tuple[str, str]] | None = None,
    k: int = 5,
) -> float:
    """Measure retrieval recall: fraction of queries where the correct
    answer substring appears in at least one of the top-k results.

    Parameters
    ----------
    mem : MemoryLayer
        A populated MemoryLayer instance.
    queries : list of (query_text, expected_substring) pairs
        Defaults to the module-level QUERIES.
    k : int
        Number of results to retrieve per query.

    Returns
    -------
    float
        Recall in [0.0, 1.0].
    """
    if queries is None:
        queries = QUERIES
    if not queries:
        return 0.0

    hits = 0
    for query_text, expected_substr in queries:
        results = mem.retrieve(query_text, k=k)
        for hit in results:
            if expected_substr.lower() in hit.text.lower():
                hits += 1
                break

    return hits / len(queries)


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

_SBERT = None


def _get_sbert():
    global _SBERT
    if _SBERT is None:
        from sentence_transformers import SentenceTransformer
        _SBERT = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT


def sbert_embed(text: str) -> torch.Tensor:
    """Embed a single text using all-MiniLM-L6-v2."""
    return _get_sbert().encode(text, convert_to_tensor=True)


def build_memory_layer(
    *,
    graph_rerank_alpha: float = 0.0,
    **kwargs: Any,
) -> MemoryLayer:
    """Create a MemoryLayer with sbert embeddings (no SOMA attached)."""
    return MemoryLayer(
        embed_fn=sbert_embed,
        embed_dim=384,
        graph_rerank_alpha=graph_rerank_alpha,
        **kwargs,
    )


def build_soma_stack(
    config: SOMAConfig | None = None,
    device: torch.device | None = None,
) -> tuple[SOMA, Any, TextEncoder]:
    """Create a SOMA + tokenizer + encoder for graph consolidation.

    Returns (soma, tokenizer, encoder).
    """
    if device is None:
        device = torch.device("cpu")
    if config is None:
        config = SOMAConfig(
            vocab_size=256,
            text_embed_dim=64,
            sensor_output_dim=64,
            max_input_tokens=128,
            initial_integrator_count=8,
            initial_associator_count=16,
            seed=42,
        )
    # Train BPE on phase-1 facts (small corpus, fast)
    corpus = PHASE1_FACTS + FLOOD_FACTS[:50]
    tokenizer = train_bpe_tokenizer(corpus, vocab_size=config.vocab_size)
    encoder = TextEncoder(tokenizer, embed_dim=config.text_embed_dim, device=device)
    soma = SOMA(config, device=device)
    return soma, tokenizer, encoder


def populate_and_measure(
    config: SOMAConfig | None = None,
    *,
    use_soma: bool = True,
    do_consolidate: bool = True,
    do_flood: bool = True,
    graph_rerank_alpha: float = 0.0,
    k: int = 5,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run the standard store-consolidate-flood-query pipeline.

    Returns a dict with recall, timing, and graph stats.
    """
    t0 = time.perf_counter()

    mem = build_memory_layer(graph_rerank_alpha=graph_rerank_alpha)

    # Store phase-1 facts
    mem.store_batch(PHASE1_FACTS)
    t_stored = time.perf_counter()

    # Optionally attach SOMA and consolidate
    consolidation_time = 0.0
    node_count = 0
    edge_count = 0
    if use_soma and config is not None:
        soma, tokenizer, encoder = build_soma_stack(config, device=device)
        mem.attach_soma(soma, tokenizer, encoder)
        if do_consolidate:
            tc = time.perf_counter()
            mem.consolidate()
            mem.stable_capture()
            consolidation_time = time.perf_counter() - tc
        node_count = len(soma.graph.nodes)
        edge_count = len(soma.graph.edges)

    # Flood with interference memories
    if do_flood:
        mem.store_batch(FLOOD_FACTS)

    # Measure recall
    t_query = time.perf_counter()
    recall = measure_recall(mem, k=k)
    query_time = time.perf_counter() - t_query

    total_time = time.perf_counter() - t0

    return {
        "recall": recall,
        "query_time_s": round(query_time, 4),
        "consolidation_time_s": round(consolidation_time, 4),
        "total_time_s": round(total_time, 4),
        "node_count": node_count,
        "edge_count": edge_count,
        "memory_count": len(mem._ids),
    }
