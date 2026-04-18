# Dietary Advisor

Integration of Large Language Models with knowledge-based dietary recommendation systems. Master's Thesis project.

## Development with Docker

1. Build docker image:
    ```
    docker compose build dietary_advisor
    ```

2. Run docker container:
    ```
    docker compose run dietary_advisor bash
    ```
    or with just
    ```
    just dc bash
    ```

### Running Tests

```bash
just test
```

### Code Style

The project uses Ruff for linting and formatting, all linters can be run with `just`:

```bash
just lint_full
just lint_full_ff  # (fast-fail mode)
just all  # (lint + tests)
just all_ff  # (lint + tests in fast-fail mode)
```
