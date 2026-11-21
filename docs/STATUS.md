# Состояние на 21.11.2026

## Что работает на телефоне

- Модуль **DolbySoundMod v1.0.54** (SELinux **Enforcing**, 6 правил из `module/sepolicy.rule`).
- Вендорный слот `/odm/lib64/soundfx/libOplusAudioxAidl.so` остаётся **стоковым** (`disable_oplus` включён) — штатный Dolby (`OplusDolby` v1.8.4) и наш per-session DSP уживаются без конфликтов.
- `AllySoundCtl` цепляет наш эффект (`a117500d-0001-4000-8000-000000000001`, `liballysoundaidl.so`) к активным сессиям выбранных приложений из `/data/adb/allysound/apps.conf` и переводит его в `PROCESSING` (`state=2`).
- В чистом Enforcing: `Line: 205 Failed = 0`, `invalid efGroup = 0`, `buffers=50` за 2 с (48 кГц), `avg_proc < 1 мс`.

## Что было исправлено

1. **Настоящая причина `Line: 205` / `invalid efGroup` в Enforcing:** `ls -lZ` показывал метку симлинка (`hal_audio_default`), а `ls -LZ` показал реальную метку inode у memfd — `u:object_r:tmpfs:s0`. Добавлены оба фикса:
   - `third_party/shim/cutils/ashmem.h`: делегирование `ashmem_create_region` системному `/system/lib64/libcutils.so` через `dlsym(RTLD_DEFAULT, ...)` (как у стоковых эффектов), `memfd_create` — fallback.
   - `module/sepolicy.rule`: `allow audioserver tmpfs file { read write open map getattr ioctl }`.
2. **Обработка буферов в `effect/ally_effect.cpp`:** убран жёсткий порог `avail_in < frame_count_ * in_channels_` (теперь потребляется ровно столько фреймов, сколько записал AudioFlinger) и исправлена маска пробуждения на `statusMQ`.
3. **WebUI (`module/webroot/app.js`):** поддержка обоих мостов KernelSU (Promise и callback) + base64 для многострочного вывода (`apps.conf`, список пакетов).
4. **Доступность модели (`vocals.onnx`):** `/data/adb` имеет права `0700 root:root`, поэтому `vocals.onnx` теперь кладётся в уже смонтированный `/vendor/lib64/soundfx/vocals.onnx`, который `audiohalservice` может читать.

## На потом

- Проверка и доводка NPU (NNAPI / `vocals.onnx`) в профилях `movie` и `podcast` (проверить в `logcat -s AllySoundNPU`, поднимается ли NNAPI в домене `hal_audio_default`).
