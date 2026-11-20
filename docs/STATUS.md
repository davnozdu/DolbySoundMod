# Состояние на 20.11.2026

## Что сейчас стоит на телефоне (рабочая конфигурация)

- Модуль **v1.0.50** (versionCode 150), сборка из коммита `8d93184`.
- Вендорный слот `/odm/lib64/soundfx/libOplusAudioxAidl.so` — **стоковый** (md5 `3d6d5691bbaa6cbf98b02e005a0d60f8`).
- Флаги: `/data/adb/allysound/disable_oplus` **есть**, `enable_oplus` **нет**, `apps.conf` пустой.
- SELinux: **Enforcing**, 5 правил из `module/sepolicy.rule`.

В таком виде звук идёт стоковым Dolby из вендорного слота, наша библиотека в тракт не
вклинивается (`dumpsys media.audio_flinger` — ноль наших эффектов, `sessions: []`).
**Это не наш DSP.** Это конфигурация, в которой телефон пригоден, пока идёт работа.

## Что выяснено (главное)

1. **Эффект на сессию приложения на этом устройстве не подключается.** Контрольный опыт:
   стоковый эффект (`41f6c0f4-5d8f-11ec-bf63-0242ac130002`) через `IFactory` к сессии
   подключается чисто, наш — отбивается сразу после `open()`:
   `handleSetConfig Line: 205 Failed`, затем `invalid efGroup` на каждом буфере.

2. **Причина — SELinux, а не дескриптор.** В permissive эффект в вендорном слоте сразу
   обрабатывает звук (`buffers=51`, `avail_in=3840`). Сужение по доменам:
   `permissive hal_audio_default` — не помогает, `permissive audioserver` — работает.

3. **Почему не видно в dmesg:** отказ идёт с `dontaudit`. Метка объясняет всё: очереди —
   это `memfd`, и SELinux помечает их **доменом создателя** (`u:r:hal_audio_default`), а не
   отдельным типом файла. У стокового эффекта область помечена как `ashmem_device`, и
   правило для `audioserver` на неё в политике есть. Лечение:

   ```
   allow audioserver hal_audio_default file { read write open map getattr ioctl }
   ```

   После этого в чистом Enforcing: `205=0`, `invalid efGroup=0`, эффект открывается
   (`Opened effect: sr=48000 in_ch=2 out_ch=2 frames=1920`).

4. `/dev/ashmem` на этом ядре — заглушка: `open` проходит, но `ASHMEM_SET_NAME`/`SET_SIZE`
   возвращают `ENOTTY`, `mmap` — `EINVAL`. Сделать настоящую ashmem-область нельзя, memfd
   остаётся единственным вариантом, поэтому правило выше обязательно.

## Оставшийся блокер

Когда эффект стоит в глобальной цепочке, клиент не присылает ему `START`/`ENABLE`:
состояние остаётся `IDLE`, `avail_in=0`, данных нет. Прозрачный обход «вход → выход» в
коде есть (ветка `else` в воркере), но он не спасает, если клиент вообще не пишет в
входную очередь — цепочка остаётся без выхода, и это слышно как полную тишину.

Дальше: разобраться, почему для эффекта из политики клиент не включает обработку
(сравнить с поведением стокового эффекта в том же слоте: у него `Enabled` приходит).

## Как быстро вернуть звук

```
su 0 rm -f /data/adb/allysound/enable_oplus
su 0 touch /data/adb/allysound/disable_oplus
su 0 printf '' > /data/adb/allysound/apps.conf
adb reboot
```

Проверка: `md5sum /odm/lib64/soundfx/libOplusAudioxAidl.so` должен быть `3d6d5691…`,
`dumpsys media.audio_flinger | grep -c Ally` — `0`.
