#!/system/bin/sh
MODDIR=${0%/*}
[ -z "$MODDIR" ] || [ "$MODDIR" = "$0" ] && MODDIR="/data/adb/modules/allysound"

DATA_DIR="/data/adb/allysound"
mkdir -p "$DATA_DIR"
[ -f "$DATA_DIR/apps.conf" ] || touch "$DATA_DIR/apps.conf"
[ -f "$DATA_DIR/device.conf" ] || echo "auto" > "$DATA_DIR/device.conf"
[ -f "$DATA_DIR/neural.conf" ] || echo "0" > "$DATA_DIR/neural.conf"
[ -f "$DATA_DIR/auto_update.conf" ] || echo "1" > "$DATA_DIR/auto_update.conf"

# Wait for Android boot to complete
while [ "$(getprop sys.boot_completed)" != "1" ]; do
    sleep 2
done

# Other modules (AutoSystemBoost mounts its own patched file over
# /odm/etc/audio_effects_config.xml) may override the config after our
# post-fs-data ran, which drops the allysound library/effect. Re-assert it once
# everything has settled and restart audioserver only if we had to.
UUID_RE="a117500d-0001-4000-8000-000000000001"
LIVE=/odm/etc/audio_effects_config.xml
[ -f "$LIVE" ] || LIVE=/vendor/etc/audio_effects_config.xml
if [ -f "$LIVE" ] && ! grep -q "$UUID_RE" "$LIVE" 2>/dev/null; then
    OUT="$DATA_DIR/live_config.xml"
    if cp -f "$LIVE" "$OUT"; then
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
        ' "$OUT" > "$OUT.new" && mv -f "$OUT.new" "$OUT"
        chmod 0644 "$OUT"
        chcon u:object_r:vendor_configs_file:s0 "$OUT" 2>/dev/null
        if mount --bind "$OUT" "$LIVE"; then
            echo "config re-asserted: $LIVE" >> "$DATA_DIR/mount.log"
            setprop ctl.restart audioserver 2>/dev/null
            sleep 3
        else
            echo "config re-assert FAILED: $LIVE" >> "$DATA_DIR/mount.log"
        fi
    fi
fi

# Background auto-update check (runs once 30s after boot if enabled)
(
    sleep 30
    if [ "$(cat "$DATA_DIR/auto_update.conf" 2>/dev/null)" = "1" ]; then
        sh "$MODDIR/bin/check_update.sh" --auto >/dev/null 2>&1
    fi
) &

# Persistent per-app audio session controller. setsid + nohup keep the
# supervisor alive on its own so it is not killed together with this script's
# process group; AllySoundCtl locks itself, so extra restarts are harmless.
nohup setsid sh -c "while true; do CLASSPATH=\"$MODDIR/bin/AllySoundCtl.dex\" /system/bin/app_process64 /system/bin AllySoundCtl >/dev/null 2>&1; sleep 3; done" >/dev/null 2>&1 &
