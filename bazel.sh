#!/usr/bin/env bash
set -euo pipefail

BAZELISK_VERSION=v1.25.0

case "$(uname -s)" in
    Darwin)
        os=darwin
        cache_dir="$HOME/Library/Caches/bazelisk"
        ;;
    Linux)
        os=linux
        cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/bazelisk"
        ;;
    *) echo 'Unsupported operating system.' >&2; exit 1 ;;
esac

case "$(uname -m)" in
    arm64|aarch64) arch=arm64 ;;
    x86_64) arch=amd64 ;;
    *) echo 'Unsupported architecture.' >&2; exit 1 ;;
esac

bazelisk="$cache_dir/bazelisk-$BAZELISK_VERSION-$os-$arch"
if [[ ! -x "$bazelisk" ]]; then
    mkdir -p "$cache_dir"
    download="$(mktemp "$cache_dir/.bazelisk.XXXXXX")"
    trap 'rm -f "$download"' EXIT
    curl -fL --retry 3 \
        "https://github.com/bazelbuild/bazelisk/releases/download/$BAZELISK_VERSION/bazelisk-$os-$arch" \
        -o "$download"
    chmod +x "$download"
    mv "$download" "$bazelisk"
    trap - EXIT
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec "$bazelisk" "$@"
