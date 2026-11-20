#!/system/bin/sh
MODDIR="/data/adb/modules/allysound"
UPDATE_URL="https://raw.githubusercontent.com/davnozdu/DolbySoundMod/main/update.json"
TMP_JSON="/data/local/tmp/allysound_update.json"
TMP_ZIP="/data/local/tmp/DolbySoundMod.zip"

CUR_CODE=$(grep -E '^versionCode=' "$MODDIR/module.prop" 2>/dev/null | cut -d= -f2 | tr -d '[:space:]')
[ -n "$CUR_CODE" ] || CUR_CODE=0

fetch_url() {
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --connect-timeout 10 -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$2" "$1"
    else
        return 1
    fi
}

fetch_url "$UPDATE_URL" "$TMP_JSON" || { echo "ERR_NETWORK"; exit 1; }

NEW_CODE=$(grep -o '"versionCode"[[:space:]]*:[[:space:]]*[0-9]*' "$TMP_JSON" | grep -o '[0-9]*' | head -n 1)
NEW_VER=$(grep -o '"version"[[:space:]]*:[[:space:]]*"[^"]*"' "$TMP_JSON" | head -n 1 | cut -d'"' -f4)
ZIP_URL=$(grep -o '"zipUrl"[[:space:]]*:[[:space:]]*"[^"]*"' "$TMP_JSON" | head -n 1 | cut -d'"' -f4)

[ -n "$NEW_CODE" ] && [ -n "$ZIP_URL" ] || { echo "ERR_PARSE"; exit 1; }

if [ "$NEW_CODE" -le "$CUR_CODE" ] 2>/dev/null; then
    echo "UP_TO_DATE:$CUR_CODE"
    rm -f "$TMP_JSON"
    exit 0
fi

fetch_url "$ZIP_URL" "$TMP_ZIP" || { echo "ERR_DOWNLOAD"; exit 1; }

if command -v ksud >/dev/null 2>&1; then
    ksud module install "$TMP_ZIP" >/dev/null 2>&1 && {
        echo "UPDATED:$NEW_VER"
        rm -f "$TMP_JSON" "$TMP_ZIP"
        exit 0
    }
fi

echo "ERR_INSTALL"
exit 1
