# SOMA Control Center — UI Guide

DearPyGUI-based desktop app that wraps training, monitoring,
visualization, and interactive chat into a single window. No CLI
commands required for normal use — launch it, click Start, and go.

## Launch

```bash
# Install the UI extras (if not already):
pip install -e ".[ui]"

# Launch with defaults:
python scripts/ui.py

# Or load a config YAML:
python scripts/ui.py --config configs/default.yaml
```

## Layout

```
+------------------------------------------------------------+
| Menu bar: Session (Start / Stop / Reset)                   |
+-------------+---------------------+-----------------------+
| Config      | [Live metrics tab]  | Chat                  |
| (left)      | [Graph view tab]    | (right)               |
|             | (center)            |                       |
+-------------+---------------------+-----------------------+
| Controls: Start/Pause/Stop/Step/Reset + session settings   |
| + checkpoint path                                          |
+------------------------------------------------------------+
```

## Typical flow

1. **Open the Controls panel.**
   - Corpus path — point at a `.txt` file (one doc per line).
   - Vocab size, snapshot interval, step budget — adjust if needed.
   - Click **Apply & rebuild** to build a fresh SOMA + tokenizer.
2. **Click Start** on the top bar (or in Controls) — training begins.
3. **Watch the Live metrics tab** — loss, curiosity, LR multiplier, graph size,
   WM occupancy, and growth-event strip update every frame.
4. **Switch to the Graph view tab** when you want to see topology. Toggle the
   "Enabled" checkbox off if it's impacting training speed. Adjust
   "Recompute every (steps)" to trade accuracy for performance.
5. **Type into the Chat panel** anytime. The panel pauses training by default
   during each exchange; untick if you want training and chat in parallel
   (both run on separate threads but share the SOMA).
6. **Save a checkpoint** from Controls. Reload later from the same pane.

## Extending the UI

Every panel is a registered plugin. To add a new one:

```python
# src/soma/ui/panels/my_panel.py
from soma.ui.registry import register_panel
from soma.ui.state import UIState

@register_panel("my_panel")
class MyPanel:
    name = "my_panel"
    display_label = "My custom panel"

    def build(self, parent, state: UIState):
        import dearpygui.dearpygui as dpg
        root = dpg.add_group(parent=parent)
        dpg.add_text("hello!", parent=root)
        return root

    def update(self, state: UIState) -> None:
        pass
```

Then import it from `src/soma/ui/panels/__init__.py`. The app will pick
it up automatically — unknown panels go into a floating "extras" tab bar
at the bottom of the window.

Panels can subscribe to any named bus channel. The built-in channels
are:

| Channel | Payload |
|---|---|
| `metrics` | per-step dict (`global_step`, `loss`, `curiosity`, `lr_multiplier`, `num_nodes`, `num_edges`, `wm_occupancy`, `episodic_entries`) |
| `growth_event` | one record when nodes or edges change (`type`, `delta_nodes`, `delta_edges`) |
| `graph_snapshot` | full topology every `snapshot_every` steps |
| `chat_response` | reserved for future use |
| `state` | controller lifecycle transitions |
| `log` | human-readable status messages |

## Troubleshooting

- **Loss barely moves.** Default `base_lr=0.001` is conservative. Bump it
  to `0.005` or `0.01` in the Config panel (Learning group), click
  "Apply & rebuild", then Start.
- **Graph view feels laggy.** Increase "Recompute every (steps)" from
  100 to 500 or disable the panel entirely. You can still get a graph
  PNG on demand via `scripts/visualize.py --png`.
- **Chat returns gibberish.** Expected at low `global_step`. Train for
  5K+ steps first.
- **Worker thread stalls.** Check the Controls status bar for an error
  message — the worker swallows exceptions and surfaces them there.

## Tests

UI logic is covered by:

- `tests/test_ui/test_bus.py` — pub/sub thread safety.
- `tests/test_ui/test_training_controller.py` — worker lifecycle, metrics publishing.
- `tests/test_ui/test_layout_engine.py` — force-directed layout stability.
- `tests/test_ui/test_registry.py` — panel plugin system.
- `tests/test_ui/test_app_smoke.py` — every panel instantiates, state wiring.

Intentionally no tests that create a DPG viewport under pytest — DPG's
`destroy_context()` teardown crashes on Windows when pytest intercepts
signals. Viewport-level verification is a manual click-through.
