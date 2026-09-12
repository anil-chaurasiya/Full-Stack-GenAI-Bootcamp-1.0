#!/usr/bin/env bash
set -euo pipefail

branch=$(git branch --show-current)

if [[ -z "$branch" ]]; then
    printf '%s\n' 'The repository is in detached HEAD state; switch to a branch first.' >&2
    exit 1
fi

printf 'Updating %s from origin/%s...\n' "$branch" "$branch"
git pull --ff-only origin "$branch"
printf 'Repository is up to date.\n'