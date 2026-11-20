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

# Background auto-update check (runs once 30s after boot if enabled)
(
    sleep 30
    if [ "$(cat "$DATA_DIR/auto_update.conf" 2>/dev/null)" = "1" ]; then
        sh "$MODDIR/bin/check_update.sh" --auto >/dev/null 2>&1
    fi
) &

# Persistent per-app audio session controller
(
    while true; do
        CLASSPATH="$MODDIR/bin/AllySoundCtl.dex" /system/bin/app_process64 /system/bin AllySoundCtl >/dev/null 2>&1
        sleep 3
    done
) &
