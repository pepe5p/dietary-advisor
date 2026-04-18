#!/bin/bash
# Thin entrypoint: hand the command to bash/dumb-init depending on shape.
# - If the first arg is a shell, exec it directly (preserves TTY).
# - If the command contains shell metacharacters, route through `bash -c`.
# - Otherwise wrap in `dumb-init` for proper PID1 signal handling.
set -e

if [[ $1 =~ ^(/bin/)?(ba)?sh$ ]]; then
    exec "$@"
elif [[ "$*" =~ ([\;\<\>]|\(|\)|\&\&|\|\|) ]]; then
    exec /bin/bash -c "$*"
fi

exec dumb-init "$@"
