# LLMScoreEngine Interactive Shell Guide

This document covers the hands-on usage of the `llmscore shell` command, including layout concepts, accessibility presets, keyboard shortcuts, configuration flags, and troubleshooting tips.

## 1. Overview

The interactive shell ships with LLMScoreEngine Beta v0.6+ and is optimized for:

- Rapidly invoking scoring actions or workflows
- Reviewing session timeline/context without leaving the terminal
- Iterating on prompts with palette search, dock quick actions, and structured output cards
- Meeting accessibility and performance requirements via explicit toggles and budgets

## 2. Quick Start

```bash
# Install dev dependencies (includes shell extras)
pip install -e ".[dev,assistant-shell]"

# Launch the shell
llmscore shell
```

On first launch the shell displays the ASCII banner, intro hints, and the help command list. Type `help` for an overview or `exit` to quit.

## 3. Configuration Basics

You can enable features through `ShellConfig` in tests or via the persisted `shell` section in your config file:

```toml
[shell]
enable_layout_v2 = true
use_output_cards = true
palette_autocomplete = true
reduced_motion = false
color_blind_mode = false
performance_monitor_enabled = true
performance_budgets.command_loop = 150.0
```

Key flags:

| Flag | Description |
|------|-------------|
| `enable_layout_v2` | Activates the multi-pane layout manager (timeline/context/dock). |
| `use_output_cards` | Renders structured Rich cards for action results. |
| `palette_autocomplete` | Adds inline chips + prompt completer for palette suggestions. |
| `reduced_motion` | Switches to static panels and removes animated transitions. |
| `color_blind_mode` | Loads a high-contrast palette for tables, dock borders, and badges. |
| `performance_monitor_enabled` | Records render/loop timings and exposes the `performance` command. |
| `performance_budgets.*` | Per-label overrides for latency budgets (ms). |

Update values interactively via `config setup` or by editing the config file managed by the CLI.

## 4. Layout & Navigation

When `enable_layout_v2` is true, the terminal splits into these panes:

- **Command Stream** – center pane listing recent commands and statuses.
- **Timeline** – configurable (left/right) pane with chronological events.
- **Context** – shows scratchpad entries and pinned notes.
- **Dock** – sticky command dock hosting quick actions and status chips.

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+K` | Open the command palette for fuzzy search |
| `Ctrl+T` | Toggle the timeline pane |
| `Ctrl+C` | Toggle the context pane |
| `Ctrl+P` | Cycle focus (Timeline → Context → Command → Dock) |
| `Ctrl+Shift+L` | Clear console (terminal-dependent) |

Focus announcements are printed whenever `Ctrl+P` cycles to a new pane, and the pane re-renders if visible.

## 5. Dock & Palette

- **Dock quick actions** refresh after every `run <action>` command using follow-up hints plus palette recommendations. Up to three shortcuts are displayed.
- **Palette autocomplete** surfaces inline toolbar chips when typing commands; press `Ctrl+K` for the full overlay.
- **Scratchpad** – `scratch add <text>` stores notes; `scratch show` renders them in the context pane.

## 6. Accessibility Presets

Use the following combinations to tailor the shell for different environments:

### Reduced Motion Mode

```toml
[shell]
reduced_motion = true
```

- Timeline renders as static text panels
- Dock output adds a "(reduced motion)" hint
- Layout avoids animated transitions

### Color-Blind Mode

```toml
[shell]
color_blind_mode = true
```

- Applies bright-white borders and alternate success/warning/error colors throughout Rich tables and Dock panels

### Focus & Screen Readers

- `Ctrl+P` ensures deterministic focus order
- Timeline/context toggles announce visibility changes and re-render content when re-enabled

## 7. Built-in Commands

Within the shell, the following verbs are available:

| Command | Description |
|---------|-------------|
| `help` | Show all commands and shortcuts |
| `actions` | List registered actions |
| `run <action>` | Execute an action (use palette or quick actions to discover names) |
| `palette [query]` | Search the action palette |
| `suggest` | Display contextual suggestions |
| `scratch add/show/remove/clear` | Manage scratchpad notes |
| `timeline` / `context` | Toggle panes (same as shortcuts) |
| `status` | Summarize pane visibility + session info |
| `performance` | Render latest budget samples (requires monitor enabled) |
| `config show/setup` | Inspect or launch the guided configuration wizard |
| `clear` | Clear the console buffer |
| `exit` / `quit` | Leave the shell |

## 8. Performance Monitoring

When `performance_monitor_enabled = true`:

- The runtime wraps command loop, command stream render, timeline render, context render, and dock render.
- Each section has a default budget (e.g., command loop 120ms). Overrides go under `performance_budgets.<label>`.
- Violations log warnings and appear in the `performance` panel:

```
> performance
Performance
-----------
command_loop: 95.33 ms / budget 120.00 ms
command_stream_render: 12.41 ms / budget 60.00 ms
```

## 9. Testing & CI Workflow

The shell is covered by targeted pytest suites:

```bash
python -m pytest tests/shell/test_phase3_runtime.py \
                 tests/shell/test_output_cards.py \
                 tests/shell/test_runtime_e2e.py
```

CI pipeline: `.github/workflows/perf-accessibility.yml` runs the same slice on PRs/pushes and installs `.[dev,assistant-shell]`. Future iterations will hook axe-core for automated accessibility audits.

## 10. Troubleshooting

| Symptom | Resolution |
|---------|------------|
| Palette toolbar missing | Ensure `palette_autocomplete = true` and prompt_toolkit is available |
| Timeline/context not rendering | Confirm `enable_layout_v2 = true` and use `Ctrl+T` / `Ctrl+C` after launch |
| `performance` command says monitor disabled | Set `performance_monitor_enabled = true` and re-launch |
| Dock throws `KeyError: 'border'` | Upgrade to Beta v0.6+, which adds safe palette merging (color-blind mode) |
| Scheduler or watcher actions unavailable | Install optional dependencies (`pip install -e .[assistant-shell]`) and re-register default actions |

## 11. Reference Links

- [Core README](../README.md)
- [UI/UX Overhaul Plan](UI_UX_OVERALL.md)
- [Shell tests](../tests/shell)

Feel free to open issues with screenshots/logs if you encounter shell regressions or have accessibility feedback.

## 12. Action Catalog

The following actions are available in the shell. Use `run <action> key=value` or implicit `<action> key=value` syntax.

### Scoring

| Action | Description | Arguments | Example |
|--------|-------------|-----------|---------|
| `score.model` | Run scoring for a single model JSON. | `model` (str), `models_dir` (str), `output_dir` (str), `quiet` (bool), `config_path` (str) | `score.model model=DeepSeek-R1` |
| `score.batch` | Run scoring for multiple models. | `models` (list), `models_dir` (str), `results_dir` (str), `quiet` (bool), `config_path` (str) | `score.batch models='["DeepSeek-R1","GPT-4"]'` |
| `score.report` | Generate CSV/HTML reports. | `models` (list), `results_dir` (str), `models_dir` (str), `output_dir` (str), `project_name` (str), `template_dir` (str), `csv` (bool), `html` (bool) | `score.report csv=true` |

### Data Management

| Action | Description | Arguments | Example |
|--------|-------------|-----------|---------|
| `data.fill` | Fill benchmark templates using API sources. | `models` (list), `models_file` (str), `config_path` (str), `template_path` (str), `output_dir` (str), `aa_key` (str), `hf_key` (str), etc. | `data.fill template_path=Templates/base.json` |
| `data.template` | Generate a model JSON template. | `model_name` (str), `overwrite` (bool) | `data.template model_name=NewModel` |
| `data.validate` | Validate model JSON files. | `models` (list), `strict` (bool) | `data.validate models='["DeepSeek-R1"]'` |

### Results Analytics

| Action | Description | Arguments | Example |
|--------|-------------|-----------|---------|
| `results.list` | List stored results by recency. | `limit` (int) | `results.list limit=5` |
| `results.show` | Show detailed result for a model. | `model` (str) | `results.show model=DeepSeek-R1` |
| `results.compare` | Compare two models. | `primary` (str), `secondary` (str), `metrics` (list) | `results.compare primary=DeepSeek-R1 secondary=GPT-4` |
| `results.export` | Export results to JSON/CSV. | `models` (list), `output_dir` (str), `format` (json/csv) | `results.export models='["DeepSeek-R1"]' output_dir=out` |
| `results.leaderboard` | Generate a leaderboard. | `sort_key` (str), `limit` (int) | `results.leaderboard sort_key=final_score` |
| `results.analyze_failures` | Analyze validation failures. | `model` (str), `minimum_score` (float) | `results.analyze_failures model=DeepSeek-R1` |
| `results.pin` | Pin a result to the session context. | `model` (str), `note` (str), `profile` (str) | `results.pin model=DeepSeek-R1 note="Top Pick"` |

### Model Registry

| Action | Description | Arguments | Example |
|--------|-------------|-----------|---------|
| `models.list` | List available model definitions. | - | `models.list` |
| `models.info` | Display model metadata. | `model` (str) | `models.info model=DeepSeek-R1` |
| `models.search` | Fuzzy search for models. | `query` (str), `limit` (int) | `models.search query=deepseek` |
| `models.tag` | Add or update model tags. | `model` (str), `tags` (list), `replace` (bool) | `models.tag model=DeepSeek-R1 tags='["verified"]'` |
| `models.sync` | Sync with external registries (placeholder). | `source` (str) | `models.sync source=hf` |

### Benchmarks

| Action | Description | Arguments | Example |
|--------|-------------|-----------|---------|
| `benchmark.list` | List benchmark collections. | - | `benchmark.list` |
| `benchmark.info` | Show benchmark metadata. | `name` (str) | `benchmark.info name=entity_benchmarks` |
| `benchmark.update` | Update a benchmark definition. | `name` (str), `source` (str) | `benchmark.update name=custom source=file.json` |
| `benchmark.validate` | Validate a benchmark file. | `name` (str) | `benchmark.validate name=custom` |

### Session & Workspace

| Action | Description | Arguments | Example |
|--------|-------------|-----------|---------|
| `session.save` | Save current session data. | `identifier` (str), `data` (dict), `profile` (str) | `session.save identifier=backup` |
| `session.load` | Load a saved session. | `identifier` (str) | `session.load identifier=backup` |
| `session.list` | List saved sessions. | `profile` (str), `limit` (int) | `session.list limit=10` |
| `session.delete` | Delete a session. | `identifier` (str) | `session.delete identifier=backup` |
| `workspace.init` | Initialize a workspace profile. | `name` (str), `workspace_path` (str), `default_session` (str) | `workspace.init name=dev workspace_path=.` |
| `workspace.list` | List workspace profiles. | - | `workspace.list` |
| `workspace.set_default_session` | Set default session for workspace. | `name` (str), `session_id` (str) | `workspace.set_default_session name=dev session_id=main` |

### Workflows

| Action | Description | Arguments | Example |
|--------|-------------|-----------|---------|
| `workflow.list` | List saved workflows. | `profile` (str) | `workflow.list` |
| `workflow.show` | Show workflow definition. | `name` (str), `profile` (str) | `workflow.show name=release-flow` |
| `workflow.import` | Import workflow from file/URL. | `path` (str), `url` (str), `profile` (str), `overwrite` (bool) | `workflow.import path=flow.json` |
| `workflow.export` | Export workflow to file. | `name` (str), `path` (str), `profile` (str) | `workflow.export name=flow path=backup.json` |
| `workflow.create` | Create new workflow. | `name` (str), `steps` (list), `description` (str), ... | `workflow.create name=new-flow steps='[...]'` |

### Automation

| Action | Description | Arguments | Example |
|--------|-------------|-----------|---------|
| `automation.watch` | Watch directory for changes. | `paths` (list), `action` (str), `recursive` (bool), `patterns` (list), ... | `automation.watch paths='["."]' action=score.model` |
| `automation.schedule` | Schedule cron job. | `action` (str), `cron` (str), `inputs` (dict) | `automation.schedule action=score.report cron="0 * * * *"` |
| `automation.list` | List active automation jobs. | - | `automation.list` |
| `automation.stop` | Stop a background job. | `identifier` (str) | `automation.stop identifier=watch::123` |

### Debugging

| Action | Description | Arguments | Example |
|--------|-------------|-----------|---------|
| `debug.inspect` | Inspect evaluation data. | `model` (str) | `debug.inspect model=DeepSeek-R1` |
