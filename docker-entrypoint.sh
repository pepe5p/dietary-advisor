#!/bin/bash
set -e

export APP=${APP:-dietary-advisor}

function log() {
  local date status message
  date=$(date -u +%Y-%m-%dT%H:%M:%S%z)
  status="$1"
  shift
  message=$(echo "$@" | sed s/$/\\n/g)
  printf '{"time":"%s","status":"%s","message":"%s"}\n' "$date" "$status" "$message"
}

if [[ $1 =~ ^(/bin/)?(ba)?sh$ ]]; then
  log INFO "First CMD argument is a shell: $1"
  log INFO "Running: exec " "${@@Q}"
  exec "$@"
elif [[ "$*" =~ ([;<>]|\(|\)|\&\&|\|\|) ]]; then
  log INFO "Shell metacharacters detected, passing CMD to bash"
  _quoted="$*"
  log INFO "Running: exec /bin/bash -c ${_quoted@Q}"
  unset _quoted
  exec /bin/bash -c "$*"
fi

# Use dumb-init to ensure proper handling of signals, zombies, etc.
# See https://github.com/Yelp/dumb-init

log INFO "Running command: dumb-init " "${@@Q}"
exec dumb-init "$@"
