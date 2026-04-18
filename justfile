set dotenv-load

PATHS_TO_LINT := "dietary_advisor tests"
TEST_PATH := "tests"
ANSWERS_FILE := ".copier/.copier-answers.copier-python-project.yml"
CONTAINER_NAME := "dietary_advisor"
WORKDIR := "/code"

[doc("Command run when 'just' is called without any arguments")]
default: help

[doc("Show this help message")]
@help:
	just --list

[group("development")]
[doc("Run all checks and tests (lints, mypy, tests...)")]
all: lint_full test

[group("development")]
[doc("Run all checks and tests, but fail on first that returns error (lints, mypy, tests...)")]
all_ff: lint_full_ff test

[group("lint")]
[doc("Run ruff lint check (code formatting)")]
ruff:
	uv run ruff check {{PATHS_TO_LINT}}
	uv run ruff format {{PATHS_TO_LINT}} --check

[group("copier")]
[doc("Update project using copier")]
copier_update answers=ANSWERS_FILE skip-answered="true":
	uv run copier update --answers-file {{answers}} \
	{{ if skip-answered == "true" { "--skip-answered" } else { "" } }}

[group("lint")]
[doc("Run fawltydeps lint check (deopendency issues)")]
deps:
	uv run fawltydeps

[group("lint")]
[doc("Run all lightweight lint checks (no mypy)")]
@lint:
	-just deps
	-just ruff

[group("lint")]
[doc("Run all lightweight lint checks, but fail on first that returns error")]
lint_ff: deps ruff

[group("lint")]
[doc("Automatically fix lint problems (only reported by ruff)")]
lint_fix:
	uv run ruff check {{PATHS_TO_LINT}} --fix
	uv run ruff format {{PATHS_TO_LINT}}

[group("lint")]
[doc("Run all lint checks and mypy")]
lint_full: lint mypy
alias full_lint := lint_full

[group("lint")]
[doc("Run all lint checks and mypy, but fail on first that returns error")]
lint_full_ff: lint_ff mypy
alias full_lint_ff := lint_full_ff

[group("lint")]
[doc("Run mypy check (type checking)")]
mypy:
	uv run mypy {{PATHS_TO_LINT}} --show-error-codes --show-traceback --implicit-reexport

[group("development")]
[doc("Run IPython with custom startup script")]
ps startup_script="ipython_startup.py":
	PYTHONSTARTUP={{ startup_script }} uv run ipython
alias ipython := ps

[group("development")]
[doc("Run non-integration tests (optionally specify file=path/to/test_file.py)")]
test file=TEST_PATH:
	uv run pytest {{file}} --durations=10

[group("docker")]
[doc("Run any just recipe inside docker (e.g. just dc test)")]
dc command *args:
	docker compose run --rm -w {{WORKDIR}} {{CONTAINER_NAME}} just {{command}} {{args}}

[group("docker")]
[doc("Build the dev/runtime docker image")]
dbuild:
	docker compose build {{CONTAINER_NAME}}

[group("docker")]
[doc("Run the CLI inside docker. Pass args, e.g.: just cli recommend L1_active_male --variant V4")]
cli *args:
	docker compose run --rm -w {{WORKDIR}} {{CONTAINER_NAME}} uv run --no-sync python -m dietary_advisor {{args}}

[group("docker")]
[doc("Run pytest inside docker (mirrors `just test` but in the container)")]
dtest file=TEST_PATH:
	docker compose run --rm -w {{WORKDIR}} {{CONTAINER_NAME}} just test {{file}}

[group("docker")]
[doc("Run full lint suite inside docker")]
dlint_full:
	docker compose run --rm -w {{WORKDIR}} {{CONTAINER_NAME}} just lint_full

[group("docker")]
[doc("Run lint + tests inside docker")]
dall:
	docker compose run --rm -w {{WORKDIR}} {{CONTAINER_NAME}} just all

[group("docker")]
[doc("Open a bash shell inside the container")]
dshell:
	docker compose run --rm -w {{WORKDIR}} {{CONTAINER_NAME}} bash

[group("development")]
[doc("Open bash console (useful when prefixed with dc, as it opens bash inside docker)")]
@bash:
	bash
