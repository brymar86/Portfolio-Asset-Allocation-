# Contributing Guidelines

## Pre-Commit Checklist

**⚠️ IMPORTANT: Before making any commit, you MUST run:**

```bash
uv run ruff check
```

All code must pass ruff linting checks before committing. This ensures code quality and consistency across the repository.

### Quick Pre-Commit Workflow

```bash
# 1. Check for linting issues
uv run ruff check

# 2. If issues are found, fix them (ruff can auto-fix many issues)
uv run ruff check --fix

# 3. Verify all checks pass
uv run ruff check

# 4. Then commit
git add .
git commit -m "Your commit message"
```

## Code Quality Standards

- All Python code must pass `ruff check` without errors
- Code should follow PEP 8 style guidelines (enforced by ruff)
- Type hints are encouraged where appropriate
- Docstrings should be included for all public functions and classes

## Development Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Install development tools (ruff is included):
   ```bash
   uv add --dev ruff
   ```

3. Run linting:
   ```bash
   uv run ruff check
   ```

## Commit Message Guidelines

- Use clear, descriptive commit messages
- Reference issue numbers if applicable
- Keep commits focused on a single change when possible

## Pull Request Process

1. Ensure all tests pass
2. Ensure `uv run ruff check` passes
3. Update documentation if needed
4. Submit pull request with clear description

