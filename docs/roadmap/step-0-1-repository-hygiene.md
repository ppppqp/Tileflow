# Step 0.1: Repository Hygiene and Developer Loop

Scope: 2-3 days.

Goal: make the project reproducible enough that every compiler experiment can be run locally with one or two commands.

Build:

- Finish package metadata in `pyproject.toml`.
- Add `Makefile` or `justfile` commands for format, lint, tests, examples, and smoke checks.
- Add `docs/notes/` and a first learning note.
- Keep CI optional for now, but design commands so CI can call the same local targets later.

Tests:

- `python3 -m compileall -q src examples tests`
- `python3 -m pytest -q` once pytest is installed.
- `PYTHONPATH=src python3 examples/vector_add.py`

References:

- `ref/Enigma-DSL/pyproject.toml`
- `../tilelang/pyproject.toml`

Learning focus:

- Python packaging.
- Editable installs.
- Reproducible compiler experiments.

Writing output:

- Blog seed: "Setting up a tiny compiler project so experiments stay honest."

