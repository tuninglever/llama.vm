#!/usr/bin/env bash
# Apply the llama.vm llama.cpp patches to a llama.cpp source tree.
#
# Usage: patches/apply-llama.cpp-patches.sh [path-to-llama.cpp]
#
# The patches are applied with `git apply` from the llama.cpp repo root, so the
# tree must be a git checkout (or at least have the files at the expected paths).
# They are idempotent-friendly: applying an already-applied patch is a no-op
# (git apply refuses cleanly, and we skip it).
set -euo pipefail

LLAMA_CPP_DIR="${1:-${LLAMA_CPP_DIR:-}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${LLAMA_CPP_DIR}" ]]; then
    echo "usage: $0 <path-to-llama.cpp>" >&2
    exit 1
fi

cd "${LLAMA_CPP_DIR}"

apply_patch() {
    local patch="$1"
    local name
    name="$(basename "${patch}")"
    if git apply --check "${patch}" 2>/dev/null; then
        echo "applying ${name}"
        git apply "${patch}"
    else
        if git apply --reverse --check "${patch}" 2>/dev/null; then
            echo "skipping ${name} (already applied)"
        else
            echo "ERROR: ${name} does not apply cleanly and is not already applied" >&2
            exit 1
        fi
    fi
}

apply_patch "${SCRIPT_DIR}/ggml-type66-kimi-q1-0-stub.patch"

echo "llama.cpp patches applied. Rebuild the ggml libs:"
echo "  cmake --build ${LLAMA_CPP_DIR}/build --target ggml-base ggml -j"
