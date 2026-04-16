"""Browser-based chat UI for a SOMA bundle.

Spins up a Gradio app so non-Python users can chat with their memory
bundle from a browser. Uses the same ``RAGSession`` + backend
abstraction as the CLI so any LLM (Ollama / OpenAI / Anthropic /
vLLM / HF) works, and shows retrieved sources inline.

Usage::

    # Start with a pre-built bundle + auto-picked LLM backend:
    python scripts/demo_web_ui.py --bundle my-brain/

    # Dry-run (no LLM — just show retrieved chunks):
    python scripts/demo_web_ui.py --bundle my-brain/ --dry-run

    # Specific backend:
    python scripts/demo_web_ui.py --bundle my-brain/ --backend ollama

    # With recall boosters:
    python scripts/demo_web_ui.py --bundle my-brain/ \\
      --hybrid-alpha 0.3 --rerank-top-n 20

Requires ``gradio`` (optional dep - ``pip install gradio``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from soma.llm import DryRunBackend, RAGSession, backend_from_env
from soma.memory import MemoryLayer


def _resolve_backend(backend_name: str, *, dry_run: bool):
    if dry_run or backend_name == "dry-run":
        return DryRunBackend()
    if backend_name == "auto":
        return backend_from_env()
    return backend_from_env(prefer=backend_name)


def _attach_reranker_if_requested(mem: MemoryLayer, rerank_top_n: int | None) -> None:
    if rerank_top_n is None:
        return
    from soma.memory.rerank import CrossEncoderReranker

    mem.attach_reranker(CrossEncoderReranker())


def build_app(
    bundle: Path,
    *,
    backend_name: str,
    dry_run: bool,
    k: int,
    hybrid_alpha: float | None,
    rerank_top_n: int | None,
) -> object:
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "demo_web_ui needs gradio. Install: pip install gradio"
        ) from exc

    mem = MemoryLayer.load(bundle)
    _attach_reranker_if_requested(mem, rerank_top_n)
    backend = _resolve_backend(backend_name, dry_run=dry_run)
    retrieve_kwargs: dict[str, object] = {}
    if hybrid_alpha is not None:
        retrieve_kwargs["hybrid_alpha"] = hybrid_alpha
    if rerank_top_n is not None:
        retrieve_kwargs["rerank_top_n"] = rerank_top_n
    session = RAGSession(
        memory=mem, llm=backend, k=k, retrieve_kwargs=retrieve_kwargs
    )

    header = (
        f"### SOMA chat — bundle `{bundle}` — {len(mem):,} entries — "
        f"LLM `{backend.name}`"
    )
    if hybrid_alpha is not None or rerank_top_n is not None:
        bits = []
        if hybrid_alpha is not None:
            bits.append(f"hybrid α={hybrid_alpha}")
        if rerank_top_n is not None:
            bits.append(f"rerank top-{rerank_top_n}")
        header += f"  —  {' · '.join(bits)}"

    def _ask(question: str, history: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
        question = (question or "").strip()
        if not question:
            return "", history
        answer = session.ask(question)
        cite_block = "\n".join(f"- {line}" for line in answer.cite_lines())
        reply = answer.text
        if cite_block:
            reply = f"{reply}\n\n**Sources:**\n{cite_block}"
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": reply},
        ]
        return "", history

    with gr.Blocks(title="SOMA chat") as app:
        gr.Markdown(header)
        chatbot = gr.Chatbot(type="messages", height=500)
        msg = gr.Textbox(placeholder="Ask a question about the bundle...", show_label=False)
        msg.submit(_ask, inputs=[msg, chatbot], outputs=[msg, chatbot])
    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument(
        "--backend",
        choices=("auto", "ollama", "openai", "anthropic", "openai-compat", "hf", "dry-run"),
        default="auto",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--hybrid-alpha", type=float, default=None)
    p.add_argument("--rerank-top-n", type=int, default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Gradio share link (public)")
    args = p.parse_args()

    if not args.bundle.exists():
        p.error(f"bundle {args.bundle} not found")

    app = build_app(
        args.bundle,
        backend_name=args.backend,
        dry_run=args.dry_run,
        k=args.k,
        hybrid_alpha=args.hybrid_alpha,
        rerank_top_n=args.rerank_top_n,
    )
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
