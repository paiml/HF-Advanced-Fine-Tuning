#!/bin/sh
# ComputeBrick profiling for fine-tuned models
# Generates BrickProfiler JSON for pmat brick-score
#
# Usage: ./brick/profile.sh [tier]
# Tiers: tiny, small (default), medium, large
#
# bashrs compliance: deterministic, idempotent, safe
# shellcheck shell=sh

set -eu

main() {
    tier="${1:-small}"

    # Validate tier input (whitelist only)
    case "${tier}" in
        tiny)   model='Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF'; target_tps='1000' ;;
        small)  model='Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF'; target_tps='788' ;;
        medium) model='bartowski/Qwen2.5-Coder-7B-Instruct-GGUF'; target_tps='400' ;;
        large)  model='bartowski/Qwen2.5-Coder-32B-Instruct-GGUF'; target_tps='150' ;;
        *)
            printf 'Unknown tier: %s\nValid tiers: tiny, small, medium, large\n' "${tier}" >&2
            return 1
            ;;
    esac

    printf '=== ComputeBrick Profiling ===\n'
    printf 'Tier: %s\nModel: %s\nTarget: %s tok/s\n\n' "${tier}" "${model}" "${target_tps}"

    # SEC010: Use literal path, create with validated structure
    if [ ! -d brick ]; then
        printf 'Error: brick/ directory does not exist\n' >&2
        return 1
    fi
    if [ ! -d brick/profiles ]; then
        mkdir -p brick/profiles
    fi

    output_file="brick/profiles/profile_${tier}.json"

    # Run profiling
    if command -v apr >/dev/null 2>&1; then
        printf 'Running apr cbtop...\n'
        if apr cbtop --tier "${tier}" --headless --json >"${output_file}" 2>&1; then
            printf '\n=== ComputeBrick Score ===\n'
            pmat brick-score --input "${output_file}" || true
        else
            printf 'apr cbtop failed\n' >&2
        fi
    else
        printf 'apr-cli not installed. Install with: cargo install apr-cli\n'
        printf 'Generating placeholder profile...\n'
        printf '{\n  "tier": "%s",\n  "model": "%s",\n  "target_tps": %s,\n  "measured_tps": null,\n  "status": "apr-cli not available"\n}\n' \
            "${tier}" "${model}" "${target_tps}" >"${output_file}"
    fi

    printf '\nProfile saved to: %s\n' "${output_file}"
}

main "${@:-}"
