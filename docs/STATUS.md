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

## 23.11.2026: нейросеть (фильмы/подкасты)

- **Шум при включении «NPU» устранён.** AudioFlinger отдаёт эффекту 1920 фреймов (15 hop'ов) за раз, а задержка конвейера была 11 hop'ов: 4 hop'а каждой пачки уходили в пассивную матрицу, звук переключался 50 раз/с. Теперь задержка подстраивается под размер пачки (≈55 мс при 1920).
- **NNAPI-драйвера на телефоне нет** — «NPU» был эмуляцией `nnapi-reference` в fp16. NNAPI используется только при реальном ускорителе, без fp16.
- **Hexagon NPU (QNN HTP):** перенос всего хребта в FP16 приводит к дрейфу состояния GRU. В отдельном тесте на телефоне удалось перенести только декодирование спектральной маски и сохранить состояние GRU на CPU. Методика и результаты в [NPU.md](NPU.md).
- **Модель разделена:** `tools/make_cpu_model.py` (запускается в CI) оставляет в ONNX только хребет (`vocals_backbone.onnx`), FFT/признаки/маска/overlap-add считаются в `dsp/vocals.cpp` (одно FFT на hop вместо трёх DFT). Совпадение с полной моделью −128 дБ (`dsp/test_vocals_split.cpp` в CI), CPU −20 %.
- **Тишина и простой:** в тишине модель не вызывается, через 10 с без звука выгружается из памяти, при звуке загружается в фоне.
- **Память:** одна сессия ORT на все эффекты, без арены; 3 приложения с нейросетью — 45 МБ вместо 88.

## Совпадение с плагином Ally Sound Lab

Без реверберации все пресеты (игры/фильмы × наушники/динамики/очки) совпадают с `decky-ally-sound/spatial.py` до −93…−135 дБ; комната: RT60 и уровень совпадают, спектр ±1 дБ выше 500 Гц.

## На потом

- Для варианта с QNN измерить затраты энергии и частоту пропусков при длительном воспроизведении.
