## Contributing

Thanks for contributing! This project uses standard Python tooling — please follow these steps to get started locally.

1. Environment

   - Create a virtualenv and install runtime deps:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   pip install -r dev-requirements.txt
   ```

2. Pre-commit hooks

   - Install hooks once per clone:

   ```bash
   pre-commit install
   pre-commit run --all-files
   ```

3. Running tests

   ```bash
   pytest -q
   ```

4. Code style

   - This repo uses Black, isort, and flake8. Please run `pre-commit run --all-files` before creating PRs.

5. Tests and CI

   - The repository contains a GitHub Actions workflow at `.github/workflows/ci.yml` which runs lint and tests on push/PR to `main`.

6. Commit messages

   - Use clear, imperative commit messages. Keep PRs small and focused.

If you need help setting up the environment locally, open an issue and we'll help.
