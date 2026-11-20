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
