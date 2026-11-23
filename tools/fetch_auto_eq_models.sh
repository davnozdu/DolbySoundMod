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
curl -fsSL --retry 3 -o "$temp/discogs-effnet-bsdynamic-1.onnx" \
    https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bsdynamic-1.onnx
verify a280825b334797cf677939db8cd5762c0392aedd0ca6415dbc1cd083f045e43c "$temp/discogs-effnet-bsdynamic-1.onnx"
mkdir -p "$dest"
# The HTP rejects the export's Identity (it only feeds the unused embeddings output).
python3 "$(dirname "$0")/prepare_effnet.py" "$temp/discogs-effnet-bsdynamic-1.onnx" "$temp/discogs-effnet-htp.onnx"
cp "$temp/yamnet.onnx" "$temp/yamnet.data" "$temp/discogs-effnet-htp.onnx" "$dest/"
