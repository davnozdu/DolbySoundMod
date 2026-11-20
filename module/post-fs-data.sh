#!/system/bin/sh
MODDIR=${0%/*}
[ -z "$MODDIR" ] || [ "$MODDIR" = "$0" ] && MODDIR="/data/adb/modules/allysound"

# Non-destructively patch audio_effects_config.xml so we never break AutoSystemBoost or stock effects.
patch_xml() {
    SRC="$1"
    DST="$2"
    [ -f "$SRC" ] || return 0
    mkdir -p "$(dirname "$DST")"

    if grep -q "a117500d-0001-4000-8000-000000000001" "$SRC" 2>/dev/null; then
        cp -f "$SRC" "$DST"
        return 0
    fi

    awk '
    /<\/libraries>/ && !lib_done {
        print "        <library name=\"allysound\" path=\"liballysoundaidl.so\"/>"
        lib_done=1
    }
    /<\/effects>/ && !eff_done {
        print "        <effect name=\"ally_sound\" library=\"allysound\" uuid=\"a117500d-0001-4000-8000-000000000001\" type=\"fe3199be-aed0-413f-87bb-11260eb63cf1\"/>"
        eff_done=1
    }
    { print }
    ' "$SRC" > "$DST"
    chmod 0644 "$DST"
}

if [ -f /odm/etc/audio_effects_config.xml ]; then
    patch_xml /odm/etc/audio_effects_config.xml "$MODDIR/odm/etc/audio_effects_config.xml"
elif [ -f /vendor/etc/audio_effects_config.xml ]; then
    patch_xml /vendor/etc/audio_effects_config.xml "$MODDIR/vendor/etc/audio_effects_config.xml"
fi
