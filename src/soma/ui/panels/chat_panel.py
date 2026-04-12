"""Interactive chat panel — send text in, see model's response.

Calls ``SOMA.interactive_session`` on the training controller's worker
thread (blocking SOMA.step on the UI thread would freeze the app). The
panel maintains its own history list; send button is disabled while a
response is in flight.

Because the controller's worker is typically running its training
loop, we issue chat messages via a thread-pool so they don't block
training — though in practice we *pause* training first (the panel's
"Pause during chat" checkbox, default True).
"""

from __future__ import annotations

import threading

try:
    import dearpygui.dearpygui as dpg

    DPG_AVAILABLE = True
except ImportError:  # pragma: no cover
    dpg = None
    DPG_AVAILABLE = False

import torch

from soma.io.text_decoder import TextDecoder
from soma.io.text_encoder import TextEncoder
from soma.system import SOMA
from soma.ui.registry import register_panel
from soma.ui.state import UIState
from soma.ui.training_controller import TrainingState

_HISTORY_CAPACITY = 50


@register_panel("chat")
class ChatPanel:
    name = "chat"
    display_label = "Chat"

    def __init__(self) -> None:
        self._root: int | str | None = None
        self._history_tag = "chat_history"
        self._input_tag = "chat_input"
        self._status_tag = "chat_status"
        self._pause_tag = "chat_pause_during"
        self._max_tokens_tag = "chat_max_tokens"
        self._lock = threading.RLock()
        self._pending_reply: str | None = None

    # ------------------------------------------------------------------
    def build(self, parent: int | str, state: UIState) -> int | str:
        if not DPG_AVAILABLE:  # pragma: no cover
            raise RuntimeError("DearPyGUI not installed; install soma[ui]")
        self._root = dpg.add_group(parent=parent, tag="panel_chat_root")

        dpg.add_text("Chat with SOMA", parent=self._root)
        with dpg.group(parent=self._root, horizontal=True):
            dpg.add_checkbox(
                label="Pause training during chat",
                default_value=True,
                tag=self._pause_tag,
            )
            dpg.add_input_int(
                label="max_output_tokens",
                default_value=16,
                min_value=1,
                min_clamped=True,
                step=0,
                width=150,
                tag=self._max_tokens_tag,
            )

        with dpg.child_window(
            height=260,
            parent=self._root,
            border=True,
            tag="chat_history_wrap",
        ):
            dpg.add_text("", tag=self._history_tag, wrap=0)

        with dpg.group(parent=self._root, horizontal=True):
            dpg.add_input_text(
                tag=self._input_tag,
                hint="type a message...",
                width=-120,
                on_enter=True,
                callback=self._on_send,
                user_data=state,
            )
            dpg.add_button(
                label="Send",
                tag="chat_send_btn",
                callback=self._on_send,
                user_data=state,
            )
        dpg.add_text("idle", tag=self._status_tag, parent=self._root)
        return self._root

    # ------------------------------------------------------------------
    def update(self, state: UIState) -> None:
        if not DPG_AVAILABLE:  # pragma: no cover
            return
        with self._lock:
            reply = self._pending_reply
            self._pending_reply = None
        if reply is not None:
            self._append_history(f"SOMA: {reply}")
            dpg.set_value(self._status_tag, "idle")

    # ------------------------------------------------------------------
    def _on_send(self, sender, app_data, user_data: UIState) -> None:  # type: ignore[no-untyped-def]
        state = user_data
        if not DPG_AVAILABLE:  # pragma: no cover
            return
        text = dpg.get_value(self._input_tag).strip()
        if not text:
            return
        dpg.set_value(self._input_tag, "")
        self._append_history(f"You: {text}")
        dpg.set_value(self._status_tag, "thinking...")

        pause = bool(dpg.get_value(self._pause_tag))
        max_tokens = int(dpg.get_value(self._max_tokens_tag))
        threading.Thread(
            target=self._chat_worker,
            args=(state, text, max_tokens, pause),
            daemon=True,
        ).start()

    def _chat_worker(
        self,
        state: UIState,
        text: str,
        max_tokens: int,
        pause: bool,
    ) -> None:
        """Runs on a dedicated thread. Talks to SOMA via the controller."""
        was_running = state.controller.state is TrainingState.RUNNING
        if pause and was_running:
            state.controller.pause()

        soma = state.controller.soma
        encoder = state.controller.encoder
        decoder = state.controller.decoder

        if soma is None or encoder is None or decoder is None:
            reply = "[no session - press Start first]"
        else:
            reply = _run_interactive(soma, encoder, decoder, text, max_tokens)

        if pause and was_running:
            state.controller.resume()
        with self._lock:
            self._pending_reply = reply

    # ------------------------------------------------------------------
    def _append_history(self, line: str) -> None:
        if not DPG_AVAILABLE:
            return
        # DPG's text widget grows up to some character limit — cap total size.
        current = dpg.get_value(self._history_tag) or ""
        new_text = (current + ("\n" if current else "") + line).splitlines()
        if len(new_text) > _HISTORY_CAPACITY:
            new_text = new_text[-_HISTORY_CAPACITY:]
        dpg.set_value(self._history_tag, "\n".join(new_text))


# ----------------------------------------------------------------------
# Worker-thread helpers
# ----------------------------------------------------------------------
def _run_interactive(
    soma: SOMA,
    encoder: TextEncoder,
    decoder: TextDecoder,
    text: str,
    max_tokens: int,
) -> str:
    """Blocking call into SOMA.interactive_session — kept out of the class so
    it's trivial to unit-test."""

    def encoder_fn(t: str) -> torch.Tensor:
        vec = encoder.encode_batch(t)
        if vec.numel() == 0:
            return torch.zeros((1, encoder.embed_dim), device=encoder.embedding.weight.device)
        return vec

    def decoder_fn(vec: torch.Tensor) -> str:
        return decoder.decode(vec)

    try:
        return soma.interactive_session(
            text,
            text_encoder=encoder_fn,
            text_decoder=decoder_fn,
            max_output_tokens=max_tokens,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return f"[error: {exc}]"
