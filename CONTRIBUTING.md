# Contributing to EduRisk AI

Thanks for your interest in contributing! This project was built as a team effort for CSL 460 — Data Mining at Bahria University.

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a virtual environment: `python -m venv venv`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a branch: `git checkout -b feature/your-feature`

## Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/ -v

# Format code
black src/ app/ tests/

# Lint
flake8 src/ app/ tests/
```

## Code Standards

- **Python 3.10+** — use type hints everywhere
- **Black** formatter — 100 char line length
- **isort** — import sorting (profile: black)
- **Docstrings** — Google style for all public functions
- **Tests** — add tests for any new functionality

## Project Structure

```
src/            # Core ML library (never import from app/)
app/            # Application layer (Gradio + FastAPI)
frontend/       # Next.js frontend
tests/          # Unit tests
docs/           # Documentation
```

## Pull Requests

- Keep PRs focused on a single change
- Include a clear description of what changed and why
- All tests must pass
- Update documentation if adding user-facing features

## Reporting Issues

Open a GitHub issue with:
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
