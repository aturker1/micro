#!/usr/bin/env bash
set -euo pipefail

BAZELISK_VERSION=v1.25.0

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'Only Linux x86_64 is supported.' >&2
    exit 1
fi

cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/bazelisk"
bazelisk="$cache_dir/bazelisk-$BAZELISK_VERSION-linux-amd64"
if [[ ! -x "$bazelisk" ]]; then
    mkdir -p "$cache_dir"
    download="$(mktemp "$cache_dir/.bazelisk.XXXXXX")"
    trap 'rm -f "$download"' EXIT
    curl -fL --retry 3 \
        "https://github.com/bazelbuild/bazelisk/releases/download/$BAZELISK_VERSION/bazelisk-linux-amd64" \
        -o "$download"
    chmod +x "$download"
    mv "$download" "$bazelisk"
    trap - EXIT
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec "$bazelisk" "$@"
