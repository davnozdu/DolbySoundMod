#!/system/bin/sh
# DolbySoundMod - post-fs-data
#
# KernelSU v3+ (and any root manager that does not mount module files) leaves the
# module's vendor/odm tree inert: scripts run, but /vendor/lib64/soundfx and
# /odm/etc never see the files. The module therefore mounts everything it needs
# itself - no metamodule, no other module, no root-manager feature.
#
# /vendor and /odm are read-only erofs, so a bind mount is the only way in:
#  * effect libs: a staging dir on /data with every stock lib re-bound into it plus
#    ours, then the whole dir is bound over /vendor/lib64/soundfx;
#  * audio_effects_config.xml: our patched copy is bound over the live file;
#  * model: lives in /data/adb/allysound so no /vendor/etc/allysound is needed.
# Labels matter: files under /data carry app_data_file, which audioserver may
# neither execute nor read, so everything we expose is chcon'ed to the label of
# the stock counterpart (vendor_file / vendor_configs_file).

MODDIR=${0%/*}
[ -z "$MODDIR" ] || [ "$MODDIR" = "$0" ] && MODDIR="/data/adb/modules/allysound"

DATA_DIR="/data/adb/allysound"
STAGE="$DATA_DIR/sfx_stage"
SFX_DIR="/vendor/lib64/soundfx"
LOG="$DATA_DIR/mount.log"

mkdir -p "$DATA_DIR" 2>/dev/null
: > "$LOG" 2>/dev/null
log() { echo "$*" >> "$LOG" 2>/dev/null; }

# Global kill switch: /data/adb/allysound/disable makes the module inert - no
# mounts, no config patches, no controller. Reboot to apply.
if [ -f "$DATA_DIR/disable" ]; then
    log "disable flag present: module is inert"
    exit 0
fi

# Non-destructively patch audio_effects_config.xml so we never break AutoSystemBoost or stock effects.
patch_xml() {
    SRC="$1"
    DST="$2"
    [ -f "$SRC" ] || return 1
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
    return 0
}

# Keep the module-relative copy (for root managers that do mount modules) and bind
# the patched file over the live path so it is effective either way.
apply_config() {
    SRC="$1"
    DST="$2"
    patch_xml "$SRC" "$DST" || return 0
    chcon u:object_r:vendor_configs_file:s0 "$DST" 2>/dev/null
    if mount --bind "$DST" "$SRC" 2>>"$LOG"; then
        log "config mounted: $SRC"
    else
        log "config bind FAILED: $SRC"
    fi
    return 0
}

# Stage the soundfx directory: stock libs re-bound, ours added on top.
stage_soundfx() {
    LIB="$MODDIR/vendor/lib64/soundfx/liballysoundaidl.so"
    if [ ! -f "$LIB" ]; then
        log "no liballysoundaidl.so in module: $LIB"
        return 1
    fi
    rm -rf "$STAGE"
    mkdir -p "$STAGE" || return 1
    # the mount root inherits this directory's label: leave it adb_data_file and the
    # audio HAL is denied access to everything below it
    chcon u:object_r:vendor_file:s0 "$STAGE" 2>/dev/null

    count_src=0
    for f in "$SFX_DIR"/*; do
        [ -f "$f" ] || continue
        b=$(basename "$f")
        # Copy, do not bind: a file bind here leaves the entry readable as 0 bytes
        # once the staging dir is bound over $SFX_DIR, and the effect HAL then
        # refuses every library in the directory.
        if ! cp -f "$f" "$STAGE/$b" 2>>"$LOG"; then
            log "copy FAILED: $b"
            return 1
        fi
        chcon u:object_r:vendor_file:s0 "$STAGE/$b" 2>/dev/null
        count_src=$((count_src + 1))
    done

    cp -f "$LIB" "$STAGE/" || return 1
    if [ -f "$MODDIR/vendor/lib64/libonnxruntime.so" ]; then
        cp -f "$MODDIR/vendor/lib64/libonnxruntime.so" "$STAGE/"
    fi
    # Stage the separator and auto EQ models next to the vendor effect library.
    for f in vocals_backbone.onnx vocals_dsp.bin yamnet.onnx yamnet.data discogs-effnet-bsdynamic-1.onnx; do
        if [ -f "$MODDIR/vendor/etc/allysound/$f" ]; then
            cp -f "$MODDIR/vendor/etc/allysound/$f" "$STAGE/$f"
            chmod 0644 "$STAGE/$f"
            chcon u:object_r:vendor_file:s0 "$STAGE/$f" 2>/dev/null
        fi
    done
    # The bundled Hexagon V81 runtime was validated on SM8850.
    QNN_SRC="$MODDIR/vendor/etc/allysound/ally-qnn"
    if [ -d "$QNN_SRC" ] && [ "$(getprop ro.soc.model)" = "SM8850" ] &&
       [ ! -f "$DATA_DIR/disable_qnn" ]; then
        if cp -R "$QNN_SRC" "$STAGE/ally-qnn" &&
           chmod 0755 "$STAGE/ally-qnn" &&
           chmod 0644 "$STAGE/ally-qnn/"* &&
           chcon -R u:object_r:vendor_file:s0 "$STAGE/ally-qnn"; then
            log "optional QNN runtime staged"
        else
            log "QNN staging failed; CPU separator available"
            rm -rf "$STAGE/ally-qnn"
        fi
    fi
    # Optional podcast model dropped in by the user; audioserver cannot read /data/adb.
    if [ -f "$DATA_DIR/speech.onnx" ]; then
        cp -f "$DATA_DIR/speech.onnx" "$STAGE/speech.onnx"
        chmod 0644 "$STAGE/speech.onnx"
        chcon u:object_r:vendor_file:s0 "$STAGE/speech.onnx" 2>/dev/null
    fi
    chcon u:object_r:vendor_file:s0 "$STAGE/liballysoundaidl.so" 2>/dev/null
    if [ -f "$STAGE/libonnxruntime.so" ]; then
        chcon u:object_r:vendor_file:s0 "$STAGE/libonnxruntime.so" 2>/dev/null
    fi

    count_dst=$(ls "$STAGE" | wc -l)
    if [ "$count_dst" -lt "$((count_src + 1))" ]; then
        log "stage incomplete: $count_dst entries, stock $count_src"
        return 1
    fi

    if ! mount --bind "$STAGE" "$SFX_DIR" 2>>"$LOG"; then
        log "soundfx dir bind FAILED"
        return 1
    fi
    log "soundfx staged: $count_dst entries (stock $count_src)"
    ls "$SFX_DIR" > "$DATA_DIR/soundfx.txt" 2>/dev/null
    return 0
}

# Older versions copied the full model into /data/adb, which audioserver cannot read;
# the separator is staged into soundfx above, so drop the 11 MB leftover.
install_model() {
    rm -f "$DATA_DIR/vocals.onnx"
    # Compiled Hexagon graphs (dsp/qnn_runtime.hpp). Emptied at every boot so a module
    # update never meets a graph prepared for older models or QNN libraries; the HAL
    # (uid audioserver, gid audio) writes them on first use.
    QNN_CACHE=/data/vendor/audio/allysound_qnn
    if [ -d /data/vendor/audio ]; then
        rm -rf "$QNN_CACHE"
        mkdir -p "$QNN_CACHE"
        chown 1041:1005 "$QNN_CACHE"
        chmod 0770 "$QNN_CACHE"
        chcon u:object_r:vendor_audio_data_file:s0 "$QNN_CACHE" 2>/dev/null
    fi
}

if [ -f /odm/etc/audio_effects_config.xml ]; then
    apply_config /odm/etc/audio_effects_config.xml "$MODDIR/odm/etc/audio_effects_config.xml"
elif [ -f /vendor/etc/audio_effects_config.xml ]; then
    apply_config /vendor/etc/audio_effects_config.xml "$MODDIR/vendor/etc/audio_effects_config.xml"
fi

# The QTI effect HAL resolves <library path=...> entries from the sku
# audio_effects.xml (that is where bundle/reverb/... are declared), so a library
# listed only in audio_effects_config.xml is reported as "can't find
# liballysoundaidl.so" and our effect never reaches the HAL's UUID list.
patch_sku_effects() {
    for SKU in /vendor/etc/audio/sku_*/*audio_effects.xml /odm/etc/audio/sku_*/*audio_effects.xml; do
        [ -f "$SKU" ] || continue
        case "$SKU" in *audio_effects_config.xml) continue;; esac
        grep -q '<libraries>' "$SKU" || continue
        OUT="$DATA_DIR/sku_$(echo "$SKU" | tr '/' '_')"
        cp -f "$SKU" "$OUT" 2>/dev/null || continue
        grep -q 'name="allysound"' "$OUT" || awk '
        /<\/libraries>/ && !done {
            print "        <library name=\"allysound\" path=\"liballysoundaidl.so\"/>"
            done=1
        }
        { print }
        ' "$OUT" > "$OUT.new" && mv -f "$OUT.new" "$OUT"
        # the HAL takes its UUID list from <effects> here, so the effect entry has
        # to be present in this file as well, not only in audio_effects_config.xml
        grep -q 'a117500d-0001-4000-8000-000000000001' "$OUT" || awk '
        /<\/effects>/ && !done {
            print "        <effect name=\"ally_sound\" library=\"allysound\" uuid=\"a117500d-0001-4000-8000-000000000001\" type=\"fe3199be-aed0-413f-87bb-11260eb63cf1\"/>"
            done=1
        }
        { print }
        ' "$OUT" > "$OUT.new" && mv -f "$OUT.new" "$OUT"
        chmod 0644 "$OUT"
        chcon u:object_r:vendor_configs_file:s0 "$OUT" 2>/dev/null
        if mount --bind "$OUT" "$SKU" 2>>"$LOG"; then
            log "sku config patched: $SKU"
        else
            log "sku config bind FAILED: $SKU"
        fi
    done
}

# Take over the Oplus "audiox" effect slot. The vendor effect HAL only ever loads
# the libraries from its own list (it ignores anything added to the config), so the
# only way in is to hand it our binary under a name it already loads.
stage_odm_soundfx() {
    SRC=/odm/lib64/soundfx
    [ -d "$SRC" ] || { log "no $SRC"; return 0; }
    if [ -f "$DATA_DIR/disable_oplus" ]; then
        log "oplus slot skipped: disable_oplus flag present"
        return 0
    fi
    # Off by default. The vendor policy attaches whatever lives in this slot to the global
    # output chain and never opens it, so our instance sat there failing every buffer with
    # "invalid efGroup" - audible as dropouts - while never producing sound. Only stage it
    # when the user explicitly asks for the experiment with the enable_oplus flag.
    if [ ! -f "$DATA_DIR/enable_oplus" ]; then
        log "oplus slot skipped: opt-in experiment (create $DATA_DIR/enable_oplus to try it)"
        return 0
    fi
    OURS="$MODDIR/vendor/lib64/soundfx/libOplusAudioxAidl.so"
    [ -f "$OURS" ] || { log "no $OURS"; return 0; }
    OSTAGE="$DATA_DIR/odmsfx_stage"
    rm -rf "$OSTAGE"
    mkdir -p "$OSTAGE" || return 1
    chcon u:object_r:vendor_file:s0 "$OSTAGE" 2>/dev/null
    ocount=0
    for f in "$SRC"/*; do
        [ -f "$f" ] || continue
        b=$(basename "$f")
        cp -f "$f" "$OSTAGE/$b" 2>>"$LOG" || return 1
        chcon u:object_r:vendor_file:s0 "$OSTAGE/$b" 2>/dev/null
        ocount=$((ocount + 1))
    done
    cp -f "$OURS" "$OSTAGE/libOplusAudioxAidl.so" || return 1
    chmod 0644 "$OSTAGE/libOplusAudioxAidl.so"
    chcon u:object_r:vendor_file:s0 "$OSTAGE/libOplusAudioxAidl.so" 2>/dev/null
    if mount --bind "$OSTAGE" "$SRC" 2>>"$LOG"; then
        log "odm soundfx staged: $ocount stock + allysound as libOplusAudioxAidl.so"
    else
        log "odm soundfx bind FAILED"
    fi
}

install_model
stage_soundfx
stage_odm_soundfx
patch_sku_effects
# KernelSU Next does not apply module sepolicy.rule files by itself on this build,
# so push them through its own tool and keep the result in the log.
if [ -f "$MODDIR/sepolicy.rule" ]; then
    KSUD=/data/adb/ksu/bin/ksud
    if [ -x "$KSUD" ]; then
        if "$KSUD" sepolicy apply "$MODDIR/sepolicy.rule" >>"$LOG" 2>&1; then
            log "sepolicy applied: $(grep -c '^allow' "$MODDIR/sepolicy.rule") rules"
        else
            log "sepolicy apply FAILED"
        fi
    else
        log "ksud not found, sepolicy not applied"
    fi
fi
log "post-fs-data done"
