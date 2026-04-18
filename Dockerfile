FROM python:3.13-slim-trixie

# System packages: dumb-init for signal handling, git for any uv VCS deps,
# wget+ca-certificates for fetching the `just` binary, build-essential for the
# few wheels (e.g. chromadb -> hnswlib) that occasionally need a C++ toolchain
# at install time on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        dumb-init \
        git \
        wget \
        ca-certificates \
        build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install `just` for the in-container task runner. Pick the right tarball
# for the current architecture so this image builds on both x86_64 and
# Apple Silicon hosts.
ENV JUST_VERSION=1.40.0
RUN ARCH="$(uname -m)" \
    && case "$ARCH" in \
            x86_64)  JUST_TARGET="x86_64-unknown-linux-musl" ;; \
            aarch64) JUST_TARGET="aarch64-unknown-linux-musl" ;; \
            *) echo "Unsupported arch: $ARCH" >&2 && exit 1 ;; \
        esac \
    && wget -qO- "https://github.com/casey/just/releases/download/${JUST_VERSION}/just-${JUST_VERSION}-${JUST_TARGET}.tar.gz" \
        | tar -xz -C /usr/local/bin just \
    && chmod +x /usr/local/bin/just

# uv for dependency + venv management.
COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /uvx /bin/

WORKDIR /code

# Cached dependency-only sync. Dev deps are included so the same image runs
# the CLI, the evaluation harness, and the test/lint suite (research-only
# project, no separate prod image).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --no-install-project --locked

ARG BUILD_COMMIT_SHA
ENV BUILD_COMMIT_SHA=${BUILD_COMMIT_SHA:-}

# Copy the application source last so editing code does not bust the
# expensive dependency layer above.
COPY . /code

# Final sync now that the project itself is in place.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Expose the CLI as a real `dietary-advisor` command on PATH so that an
# interactive shell inside the container (`just dc bash`) can invoke it as
# `dietary-advisor recommend ...` instead of the verbose
# `uv run --no-sync python -m dietary_advisor recommend ...`.
RUN printf '#!/bin/sh\nexec uv run --no-sync python -m dietary_advisor "$@"\n' \
        > /usr/local/bin/dietary-advisor \
    && chmod +x /usr/local/bin/dietary-advisor

# Persistent state (ChromaDB index, profile SQLite, USDA cache) lives under
# /code/.data, which is bind-mounted from the host via docker-compose so
# artefacts survive container restarts.
ENV DA_DATA_DIR=/code/.data \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["/code/docker-entrypoint.sh"]

# Default to printing CLI help. Override via:
#   docker compose run --rm dietary_advisor dietary-advisor <cmd>
#   just cli <cmd>
#   just dc bash   (then `dietary-advisor <cmd>` interactively)
CMD ["dietary-advisor", "--help"]
