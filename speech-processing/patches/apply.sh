#!/bin/sh
# Applies the patches in this directory to the livekit-agents installed for the python3 on
# PATH. Each patch is an upstream livekit/agents commit kept verbatim (`gh pr diff <n> --patch`),
# so it stops being needed on its own once livekit-agents is bumped to a release containing it:
# this script then reports it as already applied and skips it. Drop the patch file at that
# point, and this build step once no patch is left.
#
#   patches/apply.sh          apply every patch not applied yet
#   patches/apply.sh --check  verify only: exit 0 if every patch is applied or applies cleanly
#
# Uses git as a standalone patch tool (git apply), the only extra tool every image has.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
check=false
[ "${1:-}" = "--check" ] && check=true

site=$(python3 -c 'import pathlib, livekit.agents; print(pathlib.Path(livekit.agents.__file__).resolve().parents[2])')
version=$(python3 -c 'import importlib.metadata as m; print(m.version("livekit-agents"))')
echo "livekit-agents $version in $site"
cd "$site"

# Patch paths are a/livekit-agents/livekit/agents/...: -p2 makes them relative to site-packages,
# --include drops the hunks of files that are not installed (upstream tests).
for patch in "$here"/*.patch; do
    [ -e "$patch" ] || continue
    name=$(basename "$patch")
    if git apply --check --reverse -p2 --include='livekit/*' "$patch" 2>/dev/null; then
        echo "$name: already applied (the fix is in this livekit-agents release: drop the patch)"
    elif git apply --check -p2 --include='livekit/*' "$patch"; then
        if $check; then
            echo "$name: applies cleanly (not applied)"
        else
            git apply -p2 --include='livekit/*' "$patch"
            echo "$name: applied"
        fi
    else
        echo "$name: does not apply to livekit-agents $version, rework or drop it" >&2
        exit 1
    fi
done
