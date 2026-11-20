const DATA_DIR = '/data/adb/allysound';
const APPS_CONF = `${DATA_DIR}/apps.conf`;
const DEVICE_CONF = `${DATA_DIR}/device.conf`;
const NEURAL_CONF = `${DATA_DIR}/neural.conf`;
const REVERB_CONF = `${DATA_DIR}/reverb.conf`;
const AUTO_UPDATE_CONF = `${DATA_DIR}/auto_update.conf`;
const STATUS_JSON = `${DATA_DIR}/status.json`;
const MOD_DIR = '/data/adb/modules/allysound';

let appProfiles = {}; // { pkg: 'game' | 'movie' | 'music' | 'off' }
let deviceMode = 'auto';
let installedPkgs = [];

function execCmd(cmd) {
  return new Promise((resolve, reject) => {
    if (typeof ksu !== 'undefined' && typeof ksu.exec === 'function') {
      const cb = `cb_${Date.now()}_${Math.random().toString(36).slice(2)}`;
      window[cb] = (errno, stdout, stderr) => {
        delete window[cb];
        if (errno === 0) resolve(stdout || '');
        else reject(new Error(stderr || `exit ${errno}`));
      };
      ksu.exec(cmd, '{}', cb);
    } else {
      resolve('');
    }
  });
}

async function loadConfig() {
  try {
    const appsRaw = await execCmd(`cat "${APPS_CONF}" 2>/dev/null || true`);
    appProfiles = {};
    for (const line of appsRaw.split('\n')) {
      const t = line.trim();
      if (!t || t.startsWith('#')) continue;
      const idx = t.indexOf('=');
      if (idx > 0) {
        appProfiles[t.slice(0, idx).trim()] = t.slice(idx + 1).trim();
      }
    }
    const devRaw = (await execCmd(`cat "${DEVICE_CONF}" 2>/dev/null || echo auto`)).trim();
    deviceMode = ['auto', 'headphones', 'speakers', 'glasses'].includes(devRaw) ? devRaw : 'auto';
    const neuralRaw = (await execCmd(`cat "${NEURAL_CONF}" 2>/dev/null || echo 0`)).trim();
    document.getElementById('toggle-neural').checked = (neuralRaw === '1');
    const reverbRaw = (await execCmd(`cat "${REVERB_CONF}" 2>/dev/null || echo 0`)).trim();
    document.getElementById('toggle-reverb').checked = (reverbRaw === '1');
    const autoUpdRaw = (await execCmd(`cat "${AUTO_UPDATE_CONF}" 2>/dev/null || echo 1`)).trim();
    document.getElementById('toggle-autoupdate').checked = (autoUpdRaw !== '0');
  } catch (e) {
    console.error(e);
  }
  renderDevice();
  renderApps();
}

async function saveApps() {
  const lines = Object.entries(appProfiles)
    .map(([pkg, prof]) => `${pkg}=${prof}`)
    .join('\n');
  const b64 = btoa(unescape(encodeURIComponent(lines + (lines ? '\n' : ''))));
  await execCmd(`mkdir -p "${DATA_DIR}" && echo "${b64}" | base64 -d > "${APPS_CONF}"`);
}

async function saveDevice(mode) {
  deviceMode = mode;
  renderDevice();
  await execCmd(`mkdir -p "${DATA_DIR}" && echo "${mode}" > "${DEVICE_CONF}"`);
}

function renderDevice() {
  document.querySelectorAll('#device-selector .seg-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.dev === deviceMode);
  });
}

function renderApps() {
  const container = document.getElementById('apps-list');
  const entries = Object.entries(appProfiles);
  if (entries.length === 0) {
    container.innerHTML = '<div class="empty-state">Нажмите «+ Выбрать приложение», например YouTube, чтобы задать профиль звука.</div>';
    return;
  }

  container.innerHTML = '';
  const labels = [
    ['game', 'Игра'],
    ['movie', 'Фильм'],
    ['music', 'Музыка'],
    ['podcast', 'Подкаст / аудиокнига'],
    ['off', 'Выкл'],
  ];

  for (const [pkg, prof] of entries) {
    const row = document.createElement('div');
    row.className = 'app-row';

    const top = document.createElement('div');
    top.className = 'app-top';
    const nameEl = document.createElement('span');
    nameEl.className = 'app-pkg';
    nameEl.textContent = pkg;
    const delBtn = document.createElement('button');
    delBtn.className = 'btn-del';
    delBtn.textContent = '×';
    delBtn.onclick = async () => {
      delete appProfiles[pkg];
      renderApps();
      await saveApps();
    };
    top.append(nameEl, delBtn);

    const pills = document.createElement('div');
    pills.className = 'profile-pills';
    for (const [key, title] of labels) {
      const btn = document.createElement('button');
      btn.className = 'pill' + (prof === key ? ' active' : '');
      btn.dataset.prof = key;
      btn.textContent = title;
      btn.onclick = async () => {
        appProfiles[pkg] = key;
        renderApps();
        await saveApps();
      };
      pills.appendChild(btn);
    }

    row.append(top, pills);
    container.appendChild(row);
  }
}

async function pollStatus() {
  const badge = document.getElementById('status-badge');
  const sessEl = document.getElementById('active-sessions');
  try {
    const raw = await execCmd(`cat "${STATUS_JSON}" 2>/dev/null || true`);
    if (!raw.trim()) {
      badge.textContent = 'Ожидание сервиса';
      badge.className = 'badge badge-idle';
      return;
    }
    const st = JSON.parse(raw);
    const devNames = ['Наушники', 'Динамики', 'Очки'];
    if (st.error) {
      badge.textContent = 'Ошибка';
      badge.className = 'badge badge-err';
      sessEl.textContent = st.error;
    } else if (st.sessions && st.sessions.length > 0) {
      badge.textContent = `Активно (${st.sessions.length})`;
      badge.className = 'badge badge-ok';
      sessEl.textContent = `Выход: ${devNames[st.device] || 'Авто'} • ` +
        st.sessions.map(s => `${s.pkg} [${s.profile}]`).join(', ');
    } else {
      badge.textContent = 'Готов';
      badge.className = 'badge badge-idle';
      sessEl.textContent = `Выход: ${devNames[st.device] || 'Авто'} • Нет активных выбранных приложений`;
    }
  } catch (e) {
    badge.textContent = 'Ожидание';
    badge.className = 'badge badge-idle';
  }
}

async function openAppPicker() {
  const modal = document.getElementById('app-modal');
  const listEl = document.getElementById('modal-apps-list');
  const searchEl = document.getElementById('app-search');
  modal.classList.remove('hidden');
  searchEl.value = '';

  if (installedPkgs.length === 0) {
    listEl.innerHTML = '<div class="empty-state">Загрузка списка приложений...</div>';
    try {
      const out = await execCmd('pm list packages -3 2>/dev/null || cmd package list packages -3 2>/dev/null');
      installedPkgs = out.split('\n')
        .map(l => l.replace('package:', '').trim())
        .filter(Boolean)
        .sort();
    } catch (e) {
      listEl.innerHTML = '<div class="empty-state">Не удалось получить список пакетов</div>';
      return;
    }
  }
  renderPickerList('');
}

function guessProfile(pkg) {
  if (pkg.includes('youtube') || pkg.includes('vlc') || pkg.includes('mxtech')) return 'movie';
  if (pkg.includes('spotify') || pkg.includes('music')) return 'music';
  if (pkg.includes('podcast') || pkg.includes('audiobook') || pkg.includes('audible') || pkg.includes('bookmate')) return 'podcast';
  return 'game';
}

function renderPickerList(query) {
  const listEl = document.getElementById('modal-apps-list');
  const q = query.trim().toLowerCase();
  const filtered = installedPkgs.filter(p => !q || p.toLowerCase().includes(q)).slice(0, 100);
  if (filtered.length === 0) {
    listEl.innerHTML = '<div class="empty-state">Ничего не найдено</div>';
    return;
  }
  listEl.innerHTML = '';
  for (const pkg of filtered) {
    const chosen = !!appProfiles[pkg];
    const btn = document.createElement('button');
    btn.className = 'modal-item' + (chosen ? ' active' : '');
    btn.textContent = chosen ? `${pkg}  •  ${appProfiles[pkg]}` : pkg;
    btn.onclick = async () => {
      if (appProfiles[pkg]) {
        delete appProfiles[pkg];
      } else {
        appProfiles[pkg] = guessProfile(pkg);
      }
      await saveApps();
      renderApps();
      renderPickerList(document.getElementById('app-search').value);
    };
    listEl.appendChild(btn);
  }
}

document.querySelectorAll('#device-selector .seg-btn').forEach(btn => {
  btn.onclick = () => saveDevice(btn.dataset.dev);
});
document.getElementById('btn-pick-app').onclick = openAppPicker;
document.getElementById('btn-done-picker').onclick = () => {
  document.getElementById('app-modal').classList.add('hidden');
  renderApps();
};
document.getElementById('btn-close-modal').onclick = () => {
  document.getElementById('app-modal').classList.add('hidden');
};
document.getElementById('app-search').oninput = (e) => renderPickerList(e.target.value);

document.getElementById('toggle-neural').onchange = async (e) => {
  const v = e.target.checked ? '1' : '0';
  await execCmd(`mkdir -p "${DATA_DIR}" && echo "${v}" > "${NEURAL_CONF}"`);
};

document.getElementById('toggle-reverb').onchange = async (e) => {
  const v = e.target.checked ? '1' : '0';
  await execCmd(`mkdir -p "${DATA_DIR}" && echo "${v}" > "${REVERB_CONF}"`);
};

document.getElementById('toggle-autoupdate').onchange = async (e) => {
  const v = e.target.checked ? '1' : '0';
  await execCmd(`mkdir -p "${DATA_DIR}" && echo "${v}" > "${AUTO_UPDATE_CONF}"`);
};

document.getElementById('btn-check-update').onclick = async () => {
  const sub = document.getElementById('update-sub');
  const btn = document.getElementById('btn-check-update');
  btn.disabled = true;
  sub.textContent = 'Проверка обновлений на GitHub...';
  try {
    const out = (await execCmd(`sh "${MOD_DIR}/bin/check_update.sh"`)).trim();
    if (out.startsWith('UP_TO_DATE:')) {
      sub.textContent = 'Установлена последняя версия.';
    } else if (out.startsWith('UPDATED:')) {
      sub.textContent = `Обновлено до ${out.split(':')[1]}! Перезагрузите устройство.`;
    } else {
      sub.textContent = `Статус: ${out || 'ошибка проверки'}`;
    }
  } catch (e) {
    sub.textContent = 'Не удалось проверить обновления.';
  } finally {
    btn.disabled = false;
  }
};

loadConfig();
pollStatus();
setInterval(pollStatus, 2000);
