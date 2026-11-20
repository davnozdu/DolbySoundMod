#!/system/bin/sh
MODDIR=${0%/*}
[ -z "$MODDIR" ] || [ "$MODDIR" = "$0" ] && MODDIR="/data/adb/modules/allysound"

DATA_DIR="/data/adb/allysound"
mkdir -p "$DATA_DIR"
[ -f "$DATA_DIR/apps.conf" ] || touch "$DATA_DIR/apps.conf"
[ -f "$DATA_DIR/device.conf" ] || echo "auto" > "$DATA_DIR/device.conf"

# Wait for Android boot to complete
while [ "$(getprop sys.boot_completed)" != "1" ]; do
    sleep 2
done

# Persistent controller loop
(
    while true; do
        CLASSPATH="$MODDIR/bin/AllySoundCtl.dex" /system/bin/app_process64 /system/bin AllySoundCtl >/dev/null 2>&1
        sleep 3
    done
) &
