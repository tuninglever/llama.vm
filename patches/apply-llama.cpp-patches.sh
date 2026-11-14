#!/usr/bin/env bash
# Apply the llama.vm llama.cpp patches to a llama.cpp source tree.
#
# Usage: patches/apply-llama.cpp-patches.sh [--with-forthllm | --pager-only] [path-to-llama.cpp]
#
# By default applies all llama.vm patches: the ggml type-66 stub and the
# ForthLLM feature patch (the --forthllm-* llama-server flags + KV-cache
# compaction engine). --pager-only applies just the type-66 stub (the pager's
# minimum). --with-forthllm is the default and is accepted for compatibility.
# The LLAMA_VM_WITH_FORTHLLM env var sets the default (0 = pager-only).
#
# The patches are applied with `git apply` from the llama.cpp repo root, so the
# tree must be a git checkout (or at least have the files at the expected paths).
# They are idempotent-friendly: applying an already-applied patch is a no-op
# (git apply refuses cleanly, and we skip it).
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-}"
case "${LLAMA_VM_WITH_FORTHLLM:-1}" in
    0|no|off|false) WITH_FORTHLLM=0 ;;
    *) WITH_FORTHLLM=1 ;;
esac
CLI_FORTHLLM=""
for arg in "$@"; do
    case "${arg}" in
        --with-forthllm) CLI_FORTHLLM=1 ;;
        --pager-only)    CLI_FORTHLLM=0 ;;
        -*) echo "usage: $0 [--with-forthllm | --pager-only] <path-to-llama.cpp>" >&2; exit 1 ;;
        *) LLAMA_CPP_DIR="${arg}" ;;
    esac
done
[[ -n "${CLI_FORTHLLM}" ]] && WITH_FORTHLLM="${CLI_FORTHLLM}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${LLAMA_CPP_DIR}" ]]; then
    echo "usage: $0 [--with-forthllm | --pager-only] <path-to-llama.cpp>" >&2
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

apply_patch "${SCRIPT_DIR}/build-xcframework.sh.patch"
apply_patch "${SCRIPT_DIR}/ggml-type66-kimi-q1-0-stub.patch"

if [[ "${WITH_FORTHLLM}" == "1" ]]; then
    apply_patch "${SCRIPT_DIR}/llama-cpp-forthllm-kv-compaction.patch"
fi

echo "llama.cpp patches applied. Rebuild the ggml libs in your build dir"
echo "(build, build-hip, build-cuda, build-mlx — repeat for each one you use):"
echo "  cmake --build ${LLAMA_CPP_DIR}/<build-dir> --target ggml-base ggml -j"
if [[ "${WITH_FORTHLLM}" == "1" ]]; then
    echo "ForthLLM also applied; reconfigure and rebuild the server:"
    echo "  cmake -S ${LLAMA_CPP_DIR} -B ${LLAMA_CPP_DIR}/<build-dir> && \\"
    echo "  cmake --build ${LLAMA_CPP_DIR}/<build-dir> --target llama-server test-forthllm -j"
else
    echo "(pager-only: the --forthllm-* flags are not available)"
fi
