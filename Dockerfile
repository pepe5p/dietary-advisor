FROM python:3.13-slim-trixie

# Install compilation dependencies
RUN apt update && apt install -y \
    dumb-init \
    git \
    python3-dev \
    wget \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Install just
ENV JUST_VERSION=1.40.0
RUN wget -qO- \
    "https://github.com/casey/just/releases/download/${JUST_VERSION}/just-${JUST_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
    | tar -xz -C /usr/local/bin just

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /uvx /bin/

# Allows docker to cache installed dependencies between builds
WORKDIR /code

# Set uv environment as global instead of .venv
# ENV UV_PROJECT_ENVIRONMENT="/usr/local"
# ENV UV_SYSTEM_PYTHON=true

# Installing "standard" dependencies only, without dev. Done before getting commit sha to cache this layer
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-dev --no-install-project --locked

ARG BUILD_COMMIT_SHA
ENV BUILD_COMMIT_SHA=${BUILD_COMMIT_SHA:-}

# Installing dev dependencies only (on top of standard dependencies installed above). Triggered only for test image.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    if [ "$BUILD_COMMIT_SHA" = "localdev" ]; then \
    	uv sync --no-install-project --locked; \
    fi

# Adds our application code to the image
COPY . /code
WORKDIR /code/dietary_advisor

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$BUILD_COMMIT_SHA" = "localdev" ]; then \
    	uv sync --locked; \
    else \
    	uv sync --no-dev --locked; \
    fi

ENTRYPOINT ["/code/docker-entrypoint.sh"]

EXPOSE 80/tcp

# Runs the production server
CMD ["uv", "run", "--no-sync", "python", "main:app", "--host", "0.0.0.0", "--port", "80"]
