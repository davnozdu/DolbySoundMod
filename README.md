# DolbySoundMod (KernelSU)

Порт бинаурального 7.1 / HRTF-аудиопроцессора из Ally Sound Lab на Android в виде независимого модуля KernelSU с выбором приложений через WebUI.

## Возможности
- **Не зависит от внешних модулей** (не требует Zygisk / LSPosed) и **не ломает `AutoSystemBoost`** — `post-fs-data.sh` аккуратно дополняет текущий `audio_effects_config.xml`.
- **Выбор приложений через KernelSU WebUI**: нажмите **«+ Выбрать приложение»** (например, `com.google.android.youtube`) и задайте профиль:
  - **Игра (`game`)** — точная локализация, сухая комната (`RT60 = 0.22s`), без компрессии, только look-ahead лимитер.
  - **Фильм (`movie`)** — усиление диалогов (`+2 dB`), более объёмная комната (`RT60 = 0.40s`), левелер `-20 LUFS` + компрессор `3:1` + лимитер.
  - **Музыка (`music`)** — широкая сцена (`RT60 = 0.25s`), без акцента на центр и без компрессии динамики.
  - **Выкл (`off`)** — отключено.
- **Типы выхода**: `Авто` (определяется по `dumpsys audio`), `Наушники` (HRTF KU100), `Динамики` (CrosstalkWidener + задержка тылов), `Очки` (HRTF + подъём баса `+2 dB`).

## Сборка
Сборка выполняется автоматически в GitHub Actions (`.github/workflows/build.yml`):
1. Генерация `dsp/hrir_ku100.hpp` из `SADIE_KU100_DFC_256_order_fir_48000.sofa`.
2. Клонирование AIDL-определений из AOSP (`android.hardware.audio.effect`, `android.hardware.audio.common`, `android.hardware.common`, `android.hardware.common.fmq`, `android.media.audio.common.types`, `android.media.audio.eraser.types`) и C++-заголовков `libbinder_ndk` (в NDK их нет).
3. Компиляция AIDL в NDK C++ (`--lang=ndk --stability=vintf --min_sdk_version=35`) и сборка одного самодостаточного `liballysoundaidl.so` (`arm64-v8a`): все сгенерированные единицы трансляции + DSP + эффект, libc++ статически, из внешних зависимостей — только системные `libbinder_ndk`, `liblog`, `libm`, `libdl`, `libc`.
4. Компиляция `AllySoundCtl.dex`.
5. Упаковка в готовый `DolbySoundMod.zip` для установки в KernelSU.

## Как эффект попадает в систему
На устройствах с QTI-прошивкой (OnePlus/Oppo, этот случай) вендорный effect-HAL **не** подхватывает библиотеки, добавленные в `audio_effects*.xml`: он пишет `parseLibrary can't find …` даже про стоковые библиотеки, которые лежат в его же каталоге, и создаёт только то, что вкомпилировано в него или загружено под известным ему именем. Поэтому модуль:

1. монтирует `/odm/lib64/soundfx` как staging-каталог: стоковые библиотеки копируются, а `libOplusAudioxAidl.so` подменяется нашей сборкой (она отвечает за UUID `41f6c0f4-5d8f-11ec-bf63-0242ac130002`, то есть за тот слот, который HAL реально создаёт на медиа-сессиях);
2. дополнительно патчит `audio_effects_config.xml` и sku `audio_effects.xml` (на случай других прошивок, где конфиг работает);
3. ставит свою копию библиотеки в `/vendor/lib64/soundfx` (staging с копированием, а не bind-файлами — bind-файл читается как 0 байт после bind каталога).

Отключить подмену Oplus-слота, не удаляя модуль:

```
su -c 'touch /data/adb/allysound/disable_oplus'
# затем перезагрузка
```

Убрать обратно: `rm /data/adb/allysound/disable_oplus` и перезагрузить.
