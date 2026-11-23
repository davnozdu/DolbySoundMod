#!/usr/bin/env bash
set -euo pipefail

dest=${1:?Provide the model output directory}
temp=$(mktemp -d)
trap 'rm -rf -- "$temp"' EXIT

verify() {
    local expected=$1 file=$2 actual
    if command -v sha256sum >/dev/null; then
        actual=$(sha256sum < "$file")
    else
        actual=$(shasum -a 256 < "$file")
    fi
    if [[ ${actual%% *} != "$expected" ]]; then
        printf 'SHA-256 mismatch: %s\n' "$file" >&2
        exit 1
    fi
}

curl -fsSL --retry 3 -o "$temp/yamnet.zip" \
    https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/yamnet/releases/v0.59.0/yamnet-onnx-float.zip
verify bda628207418bc59b6a371aef6180b53367707eaf7b8b8a7c683417000d1eea4 "$temp/yamnet.zip"
unzip -jq "$temp/yamnet.zip" 'yamnet-onnx-float/yamnet.onnx' 'yamnet-onnx-float/yamnet.data' -d "$temp"
curl -fsSL --retry 3 -o "$temp/msd-musicnn-1.onnx" \
    https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.onnx
verify 49668ffec47e52e94b96f45930bb46a28a1368d4bdfb5c05378fa834aca616e1 "$temp/msd-musicnn-1.onnx"
mkdir -p "$dest"
cp "$temp/yamnet.onnx" "$temp/yamnet.data" "$temp/msd-musicnn-1.onnx" "$dest/"
