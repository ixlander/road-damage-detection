# Contributing

1. Install dependencies:
   - `pip install -r requirements-dev.txt -c constraints.txt`
2. Enable hooks:
   - `pre-commit install`
3. Run checks locally:
   - `ruff check .`
   - `mypy src apps api demo datasets`
   - `pytest`

Before submitting:
- preserve class-map invariant (`0 crack`, `1 pothole`)
- avoid duplicate logic; use shared modules
- keep API compatibility or explicitly version changes
