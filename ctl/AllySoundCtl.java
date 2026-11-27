import android.media.AudioAttributes;
import android.media.AudioDeviceInfo;
import android.media.audiofx.AudioEffect;
import android.os.Binder;
import android.os.FileObserver;
import android.os.IBinder;
import android.os.Parcel;
import android.os.RemoteException;
import android.os.SystemClock;
import android.util.Log;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.nio.channels.FileLock;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.UUID;

public class AllySoundCtl {
    private static final UUID TYPE_LOUDNESS = UUID.fromString("fe3199be-aed0-413f-87bb-11260eb63cf1");
    private static final UUID IMPL_ALLY = UUID.fromString("a117500d-0001-4000-8000-000000000001");
    private static final String BASE_DIR = "/data/adb/allysound";
    private static final String APPS_CONF = BASE_DIR + "/apps.conf";
    private static final String DEVICE_CONF = BASE_DIR + "/device.conf";
    private static final String NEURAL_CONF = BASE_DIR + "/neural.conf";
    private static final String REVERB_CONF = BASE_DIR + "/reverb.conf";
    private static final String REVERB_MUSIC_CONF = BASE_DIR + "/reverb_music.conf";
    private static final String REVERB_MOVIE_CONF = BASE_DIR + "/reverb_movie.conf";
    private static final String REVERB_PODCAST_CONF = BASE_DIR + "/reverb_podcast.conf";
    private static final String SCENE_MUSIC_CONF = BASE_DIR + "/scene_music.conf";
    private static final String SCENE_MOVIE_CONF = BASE_DIR + "/scene_movie.conf";
    private static final String SCENE_PODCAST_CONF = BASE_DIR + "/scene_podcast.conf";
    private static final String UPMIX_CONF = BASE_DIR + "/upmix.conf";
    private static final String AUTO_EQ_CONF = BASE_DIR + "/auto_eq.conf";
    private static final String STATUS_JSON = BASE_DIR + "/status.json";

    private static class SessionEffect {
        AudioEffect effect;
        String pkg;
        int profile;
        int neural;
        int reverb;
        int upmix;
        int autoEq;
        int sceneAdapter;
        int device;
        long releaseAt;
        boolean modelsPaused;
    }

    private static final Map<Integer, SessionEffect> active = new HashMap<>();
    private static final Map<Integer, Long> failedAt = new HashMap<>();
    private static final long SESSION_GRACE_MS = 15000;
    private static final long MODEL_PAUSE_MS = 500;
    private static final long ROUTE_CONFIRM_MS = 250;
    private static final int MODELS_RESUME = 15000;
    private static final int MODELS_PAUSE = 15001;
    private static final long ACTIVE_REFRESH_MS = 1500;
    private static final long FALLBACK_IDLE_MS = 5000;

    private static int parseProfile(String s) {
        if (s == null) return 0;
        switch (s.trim().toLowerCase()) {
            case "game": return 1;
            case "movie": return 2;
            case "music": return 3;
            case "podcast":
            case "speech": return 4;
            default: return 0;
        }
    }

    private static String profileName(int p) {
        switch (p) {
            case 1: return "game";
            case 2: return "movie";
            case 4: return "podcast";
            case 3: return "music";
            default: return "off";
        }
    }

    private static int parseDevice(String s) {
        if (s == null) return 0;
        switch (s.trim().toLowerCase()) {
            case "speakers": return 1;
            case "glasses": return 2;
            default: return 0; // headphones
        }
    }

    private static Map<String, Integer> readAppsConf() {
        Map<String, Integer> map = new HashMap<>();
        File f = new File(APPS_CONF);
        if (!f.exists()) return map;
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;
                int eq = line.indexOf('=');
                if (eq <= 0) continue;
                String pkg = line.substring(0, eq).trim();
                int prof = parseProfile(line.substring(eq + 1));
                if (prof > 0) map.put(pkg, prof);
            }
        } catch (Exception ignored) {}
        return map;
    }

    private static String readDeviceMode() {
        File f = new File(DEVICE_CONF);
        if (!f.exists()) return "auto";
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String line = br.readLine();
            return line != null ? line.trim().toLowerCase() : "auto";
        } catch (Exception e) {
            return "auto";
        }
    }

    private static int readToggle(String path) {
        return readToggleDefault(path, 0);
    }

    private static int readToggleDefault(String path, int def) {
        File f = new File(path);
        if (!f.exists()) return def;
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String line = br.readLine();
            return (line != null && "1".equals(line.trim())) ? 1 : 0;
        } catch (Exception e) {
            return def;
        }
    }

    private static Method getMediaDevices;
    private static Method getDeviceType;
    private static Method getDeviceName;
    private static final AudioAttributes MEDIA_ATTRIBUTES = new AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA).build();

    private static boolean isGlasses(String name) {
        String low = name.toLowerCase(java.util.Locale.ROOT);
        return low.contains("xreal") || low.contains("nreal") ||
               low.contains("viture") || low.contains("rokid");
    }

    private static int detectOutputDevice(String mode) {
        if (!"auto".equals(mode)) return parseDevice(mode);
        try {
            if (getMediaDevices == null) {
                Class<?> deviceClass = Class.forName("android.media.AudioDeviceAttributes");
                getDeviceType = deviceClass.getMethod("getType");
                getDeviceName = deviceClass.getMethod("getName");
                getMediaDevices = Class.forName("android.media.AudioSystem")
                        .getMethod("getDevicesForAttributes", AudioAttributes.class, boolean.class);
            }
            List<?> devices = (List<?>) getMediaDevices.invoke(null, MEDIA_ATTRIBUTES, true);
            boolean needsDeviceName = false;
            for (Object item : devices) {
                switch ((Integer) getDeviceType.invoke(item)) {
                    case AudioDeviceInfo.TYPE_BLUETOOTH_A2DP:
                    case AudioDeviceInfo.TYPE_BLUETOOTH_SCO:
                    case AudioDeviceInfo.TYPE_BLE_HEADSET:
                    case AudioDeviceInfo.TYPE_BLE_BROADCAST:
                    case AudioDeviceInfo.TYPE_WIRED_HEADSET:
                    case AudioDeviceInfo.TYPE_WIRED_HEADPHONES:
                    case AudioDeviceInfo.TYPE_HEARING_AID:
                    case AudioDeviceInfo.TYPE_USB_HEADSET:
                        return 0;
                    case AudioDeviceInfo.TYPE_USB_DEVICE:
                    case AudioDeviceInfo.TYPE_USB_ACCESSORY:
                    case AudioDeviceInfo.TYPE_HDMI:
                    case AudioDeviceInfo.TYPE_HDMI_ARC:
                    case AudioDeviceInfo.TYPE_HDMI_EARC:
                        if (isGlasses((String) getDeviceName.invoke(item))) return 2;
                        needsDeviceName = true;
                        break;
                    default:
                        break;
                }
            }
            if (!devices.isEmpty() && !needsDeviceName) return 1;
        } catch (Exception ignored) {
            // Older vendor frameworks may expose only the dump.
        }
        return detectOutputDeviceFromDump();
    }

    private static int detectOutputDeviceFromDump() {
        // Use the route of STREAM_MUSIC ("- STREAM_MUSIC:" ... "Devices: speaker(2)").
        // Scanning the whole dump for "a2dp" matched Bluetooth history lines
        // ("A2DP device ... made unavailable") and put the headphone HRTF on the
        // phone speaker.
        String route = null;
        boolean sawGlasses = false;
        Process p = null;
        try {
            p = Runtime.getRuntime().exec(new String[]{"dumpsys", "audio"});
            BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String line;
            boolean inMusic = false;
            while ((line = br.readLine()) != null) {
                String low = line.toLowerCase();
                if (isGlasses(low)) {
                    sawGlasses = true;
                }
                if (low.trim().startsWith("- stream_music:")) {
                    inMusic = true;
                    continue;
                }
                if (inMusic && route == null && low.trim().startsWith("devices:")) {
                    route = low.trim().substring("devices:".length()).trim();
                    inMusic = false;
                }
            }
        } catch (Exception e) {
            return -1;
        } finally {
            if (p != null) p.destroy();
        }
        if (route == null || route.isEmpty()) return -1;
        if (route.contains("a2dp") || route.contains("bt_sco") || route.contains("ble_headset") ||
            route.contains("ble_broadcast") || route.contains("headset") || route.contains("headphone") ||
            route.contains("hearing_aid")) {
            return 0;
        }
        if ((route.contains("usb") || route.contains("hdmi")) && sawGlasses) return 2;
        if (route.contains("usb")) return 0; // USB DAC / dongle
        return 1; // speaker, earpiece, anything else
    }

    private static int intAfter(String line, String key) {
        int at = line.indexOf(key);
        if (at < 0) return -1;
        int i = at + key.length();
        while (i < line.length() && !Character.isDigit(line.charAt(i))) i++;
        int start = i;
        while (i < line.length() && Character.isDigit(line.charAt(i))) i++;
        if (i == start) return -1;
        try {
            return Integer.parseInt(line.substring(start, i));
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    /** Package name of a running process: for an app its cmdline is the package name. */
    private static String packageOfPid(int pid) {
        try {
            byte[] raw = java.nio.file.Files.readAllBytes(java.nio.file.Paths.get("/proc/" + pid + "/cmdline"));
            String all = new String(raw, "UTF-8").replace('\0', '\n').trim();
            if (all.isEmpty()) return null;
            String first = all.split("\n")[0].trim();
            return first.isEmpty() ? null : first;
        } catch (Exception e) {
            return null;
        }
    }

    private static void writeDebug(StringBuilder log, Map<Integer, String> sessions) {
        if (log == null) return;
        try {
            log.append("sessions=").append(sessions).append('\n');
            java.io.FileOutputStream out = new java.io.FileOutputStream(BASE_DIR + "/ctl_debug.txt");
            out.write(log.toString().getBytes("UTF-8"));
            out.close();
        } catch (Exception ignored) {}
    }

    /**
     * Sessions AudioFlinger is currently mixing audio for.
     *
     * The flinger's own "Global session refs" table is the obvious source but it is empty on
     * this device whenever the track is on an offload or direct output, and it also lists our
     * own AudioEffect as a "root" client of the very session we are about to attach to. The
     * audio service keeps the authoritative list of started tracks (with their session ids and
     * owning pid), so that is the primary source; the flinger table is only the fallback.
     */
    private static Map<Integer, String> readActiveSessions() {
        Map<Integer, String> sessions = new HashMap<>();
        StringBuilder log = new java.io.File(BASE_DIR + "/debug").exists() ? new StringBuilder() : null;
        boolean hasPlaybackMonitor = false;
        try {
            Process p = Runtime.getRuntime().exec(new String[]{"dumpsys", "audio"});
            BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String line;
            while ((line = br.readLine()) != null) {
                if (line.startsWith("### Playback activity")) hasPlaybackMonitor = true;
                if (!line.contains("state:started")) continue;
                if (log != null) log.append("hit: ").append(line).append('\n');
                int sid = intAfter(line, "sessionId:");
                // "u/pid:10206/6523" is uid/pid; the pid is what /proc knows about. Search for
                // the separator after the colon: the key itself starts with "u/".
                int upid = line.indexOf("u/pid:");
                int pid = -1;
                if (upid >= 0) {
                    int colon = line.indexOf(':', upid);
                    int slash = (colon > 0) ? line.indexOf('/', colon) : -1;
                    if (slash > 0) {
                        int i = slash + 1;
                        int start = i;
                        while (i < line.length() && Character.isDigit(line.charAt(i))) i++;
                        if (i > start) {
                            try {
                                pid = Integer.parseInt(line.substring(start, i));
                            } catch (NumberFormatException ignored) {}
                        }
                    }
                }
                if (sid <= 0 || pid <= 0) continue;
                String pkg = packageOfPid(pid);
                if (log != null) log.append("parsed sid=").append(sid).append(" pid=").append(pid)
                        .append(" pkg=").append(pkg).append('\n');
                if (pkg != null && pkg.contains(".") && !sessions.containsKey(sid)) {
                    sessions.put(sid, pkg);
                }
            }
            p.destroy();
        } catch (Exception e) {
            if (log != null) log.append("audio dump failed: ").append(e).append('\n');
        }
        if (hasPlaybackMonitor || !sessions.isEmpty()) {
            writeDebug(log, sessions);
            return sessions;
        }

        try {
            Process p = Runtime.getRuntime().exec(new String[]{"dumpsys", "media.audio_flinger"});

            BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String line;
            boolean inGlobal = false;
            while ((line = br.readLine()) != null) {
                if (line.contains("Global session refs:")) {
                    inGlobal = true;
                    continue;
                }
                if (inGlobal) {
                    String trimmed = line.trim();
                    if (trimmed.isEmpty() || !Character.isDigit(trimmed.charAt(0))) {
                        if (trimmed.startsWith("session")) continue;
                        break;
                    }
                    String[] parts = trimmed.split("\\s+");
                    if (parts.length >= 5) {
                        try {
                            int sid = Integer.parseInt(parts[0]);
                            String pkg = parts[4];
                            // One session can list several clients: our own AudioEffect shows up
                            // as "root" next to the app. Take the last real package name, never
                            // let a bare process name overwrite it, or the controller releases
                            // the effect it has just created and starts over every cycle.
                            String prev = sessions.get(sid);
                            if (prev == null || (!prev.contains(".") && pkg.contains("."))) {
                                sessions.put(sid, pkg);
                            }
                        } catch (NumberFormatException ignored) {}
                    }
                }
            }
            p.destroy();  // an early break leaves dumpsys blocked on a full pipe;
                          // waitFor() would then hang the whole control loop
        } catch (Exception e) {
            if (log != null) log.append("flinger dump failed: ").append(e).append('\n');
        }
        writeDebug(log, sessions);
        return sessions;
    }

    private static final int PLAYBACK_CHANGED = 1;
    private static final int CONFIG_CHANGED = 2;
    private static final int AUDIO_SERVICE_DIED = 4;
    private static final int PLAYER_STATE_STARTED = 2;
    private static final String PLAYBACK_DISPATCHER = "android.media.IPlaybackConfigDispatcher";

    private static class ControllerEvents extends Binder {
        private int pending = CONFIG_CHANGED;

        ControllerEvents() {
            attachInterface(null, PLAYBACK_DISPATCHER);
        }

        @Override
        protected boolean onTransact(int code, Parcel data, Parcel reply, int flags)
                throws RemoteException {
            // dispatchPlaybackConfigChange is the first and only AIDL method.
            if (code != FIRST_CALL_TRANSACTION) return super.onTransact(code, data, reply, flags);
            data.enforceInterface(PLAYBACK_DISPATCHER);
            signal(PLAYBACK_CHANGED);
            return true;
        }

        synchronized void signal(int event) {
            pending |= event;
            notifyAll();
        }

        synchronized int await(long deadline) throws InterruptedException {
            while (pending == 0) {
                if (deadline == Long.MAX_VALUE) {
                    wait();
                } else {
                    long remaining = deadline - SystemClock.elapsedRealtime();
                    if (remaining <= 0) break;
                    wait(remaining);
                }
            }
            int events = pending;
            pending = 0;
            return events;
        }
    }

    private static Object audioService;
    private static Method getPlaybackConfigs;
    private static Method getPlayerState;
    private static Method getSessionId;
    private static Method getClientPid;

    private static boolean connectAudioService(ControllerEvents events) {
        try {
            Class<?> serviceManager = Class.forName("android.os.ServiceManager");
            IBinder binder = (IBinder) serviceManager.getMethod("getService", String.class).invoke(null, "audio");
            Class<?> stub = Class.forName("android.media.IAudioService$Stub");
            audioService = stub.getMethod("asInterface", IBinder.class).invoke(null, binder);
            Class<?> type = Class.forName("android.media.AudioPlaybackConfiguration");
            getPlayerState = type.getMethod("getPlayerState");
            getSessionId = type.getMethod("getSessionId");
            getClientPid = type.getMethod("getClientPid");
            getPlaybackConfigs = audioService.getClass().getMethod("getActivePlaybackConfigurations");
            binder.linkToDeath(() -> events.signal(AUDIO_SERVICE_DIED), 0);

            Class<?> dispatcher = Class.forName(PLAYBACK_DISPATCHER);
            Class<?> dispatcherStub = Class.forName(PLAYBACK_DISPATCHER + "$Stub");
            Object callback = dispatcherStub.getMethod("asInterface", IBinder.class).invoke(null, events);
            audioService.getClass().getMethod("registerPlaybackCallback", dispatcher)
                    .invoke(audioService, callback);
            return true;
        } catch (Exception e) {
            Log.w("AllySoundCtl", "Playback callbacks unavailable; using polling", e);
            return false;
        }
    }

    private static Map<Integer, String> readSessionsFast() throws Exception {
        Map<Integer, String> sessions = new HashMap<>();
        List<?> configs = (List<?>) getPlaybackConfigs.invoke(audioService);
        if (configs == null) return sessions;
        for (Object config : configs) {
            int state = (Integer) getPlayerState.invoke(config);
            if (state != PLAYER_STATE_STARTED) continue;
            int sid = (Integer) getSessionId.invoke(config);
            int pid = (Integer) getClientPid.invoke(config);
            // An anonymized list (pid -1 for a playing track) is useless: let the
            // caller fall back to dumpsys for good.
            if (pid <= 0) throw new IllegalStateException("anonymized playback configs");
            if (sid <= 0) continue;
            String pkg = packageOfPid(pid);
            if (pkg != null && pkg.contains(".") && !sessions.containsKey(sid)) {
                sessions.put(sid, pkg);
            }
        }
        return sessions;
    }

    private static AudioEffect createAllyEffect(int sessionId) throws Exception {
        Constructor<AudioEffect> ctor = AudioEffect.class.getDeclaredConstructor(
            UUID.class, UUID.class, int.class, int.class);
        ctor.setAccessible(true);
        // Control experiment: /data/adb/allysound/stock_test makes the controller attach the
        // vendor effect instead of ours. That tells us whether the status-queue handshake for a
        // session effect works at all on this device, or whether it is our library that the
        // client rejects.
        if (new java.io.File(BASE_DIR + "/stock_test").exists()) {
            UUID oplus = UUID.fromString("41f6c0f4-5d8f-11ec-bf63-0242ac130002");
            return ctor.newInstance(oplus, oplus, 0, sessionId);
        }
        return ctor.newInstance(TYPE_LOUDNESS, IMPL_ALLY, 0, sessionId);
    }

    private static void setAllyParam(AudioEffect fx, int profile, int neural, int reverb,
                                     int upmix, int autoEq, int sceneAdapter, int device) throws Exception {
        int flags = sceneAdapter * 4 + autoEq * 2 + upmix;
        int code = 50000 + flags * 10000 + profile * 1000 + neural * 100 + reverb * 10 + device;
        sendGainCode(fx, code);
    }

    private static void setModelsPaused(SessionEffect session, boolean paused) throws Exception {
        if (session.modelsPaused == paused) return;
        sendGainCode(session.effect, paused ? MODELS_PAUSE : MODELS_RESUME);
        session.modelsPaused = paused;
    }

    private static void sendGainCode(AudioEffect fx, int code) throws Exception {
        Method setParam = AudioEffect.class.getDeclaredMethod("setParameter", int.class, int.class);
        setParam.setAccessible(true);
        int result = (Integer) setParam.invoke(fx, 0 /* PARAM_TARGET_GAIN_MB */, code);
        if (result != AudioEffect.SUCCESS) {
            throw new IllegalStateException("AudioEffect.setParameter returned " + result);
        }
    }

    private static void releaseEffect(AudioEffect effect) {
        try {
            effect.setEnabled(false);
        } catch (Exception ignored) {}
        try {
            effect.release();
        } catch (Exception ignored) {}
    }

    private static String writeStatus(int device, String error, String previous) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"running\":true,\"device\":").append(device);
        if (error != null) {
            sb.append(",\"error\":\"").append(error.replace("\"", "'")).append("\"");
        }
        sb.append(",\"sessions\":[");
        boolean first = true;
        for (Map.Entry<Integer, SessionEffect> e : active.entrySet()) {
            if (!first) sb.append(",");
            first = false;
            SessionEffect se = e.getValue();
            sb.append("{\"id\":").append(e.getKey())
              .append(",\"pkg\":\"").append(se.pkg).append("\"")
              .append(",\"profile\":\"").append(profileName(se.profile)).append("\"")
              .append(",\"reverb\":").append(se.reverb != 0)
              .append(",\"upmix\":").append(se.upmix != 0)
              .append(",\"auto_eq\":").append(se.autoEq != 0)
              .append(",\"scene_adapter\":").append(se.sceneAdapter != 0)
              .append(",\"neural\":").append(se.neural != 0)
              .append(",\"paused\":").append(se.modelsPaused).append("}");
        }
        sb.append("]}");
        String status = sb.toString();
        if (status.equals(previous)) return previous;
        File file = new File(STATUS_JSON);
        File temp = new File(STATUS_JSON + ".new");
        try {
            try (FileWriter fw = new FileWriter(temp)) {
                fw.write(status);
            }
            Files.move(temp.toPath(), file.toPath(),
                    StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            return status;
        } catch (Exception e) {
            Log.w("AllySoundCtl", "Could not write status", e);
            return previous;
        }
    }

    public static void main(String[] args) {
        new File(BASE_DIR).mkdirs();
        // Hold the lock for the whole controller lifetime, including asynchronous callbacks.
        try (RandomAccessFile lockFile = new RandomAccessFile(BASE_DIR + "/ctl.lock", "rw");
             FileLock lock = lockFile.getChannel().tryLock()) {
            if (lock != null) runController();
        } catch (Exception e) {
            Log.e("AllySoundCtl", "Controller stopped", e);
        }
        // app_process also runs a Binder thread pool. Terminate it so service.sh can restart us.
        System.exit(0);
    }

    private static void runController() {
        ControllerEvents events = new ControllerEvents();
        FileObserver observer = new FileObserver(new File(BASE_DIR), FileObserver.CLOSE_WRITE |
                FileObserver.MOVED_TO | FileObserver.MOVED_FROM | FileObserver.DELETE) {
            @Override
            public void onEvent(int event, String path) {
                if (path != null && path.endsWith(".conf")) events.signal(CONFIG_CHANGED);
            }
        };
        observer.startWatching();
        boolean playbackCallbacks = connectAudioService(events);
        if (playbackCallbacks) Log.i("AllySoundCtl", "Watching playback and configuration changes");

        String lastError = null;
        String lastStatus = null;
        String deviceMode = "auto";
        long nextRefresh = 0;
        long deadline = 0;
        long candidateSince = 0;
        int routeCandidate = -1;
        Map<String, Integer> apps = new HashMap<>();
        int device = 1, neural = 0, upmix = 0, autoEq = 0;
        int reverbMusic = 1, reverbMovie = 0, reverbPodcast = 0;
        int sceneMusic = 1, sceneMovie = 0, scenePodcast = 0;
        while (true) {
            try {
                int changes = events.await(deadline);
                if ((changes & AUDIO_SERVICE_DIED) != 0) break;
                final boolean configChanged = (changes & CONFIG_CHANGED) != 0;
                if (configChanged) {
                    apps = readAppsConf();
                    deviceMode = readDeviceMode();
                    neural = readToggle(NEURAL_CONF);
                    reverbMusic = readToggleDefault(REVERB_MUSIC_CONF, 1);
                    reverbMovie = readToggleDefault(REVERB_MOVIE_CONF, readToggle(REVERB_CONF));
                    reverbPodcast = readToggleDefault(REVERB_PODCAST_CONF, 0);
                    sceneMusic = readToggleDefault(SCENE_MUSIC_CONF, 1);
                    sceneMovie = readToggleDefault(SCENE_MOVIE_CONF, 0);
                    scenePodcast = readToggleDefault(SCENE_PODCAST_CONF, 0);
                    upmix = readToggle(UPMIX_CONF);
                    autoEq = readToggle(AUTO_EQ_CONF);
                }
                Map<Integer, String> sessions = null;
                if (getPlaybackConfigs != null) {
                    try {
                        sessions = readSessionsFast();
                    } catch (Exception e) {
                        getPlaybackConfigs = null;
                        playbackCallbacks = false;
                        Log.w("AllySoundCtl", "Playback configurations unavailable; using dumpsys", e);
                    }
                }
                if (sessions == null) sessions = readActiveSessions();
                for (Iterator<String> packages = sessions.values().iterator(); packages.hasNext();) {
                    if (!apps.containsKey(packages.next())) packages.remove();
                }
                failedAt.keySet().retainAll(sessions.keySet());

                final long now = SystemClock.elapsedRealtime();
                final boolean watchingPlayback = !sessions.isEmpty();
                if (configChanged || (watchingPlayback && now >= nextRefresh)) {
                    int detected = detectOutputDevice(deviceMode);
                    nextRefresh = now + ACTIVE_REFRESH_MS;
                    if (detected < 0 || detected == device) {
                        routeCandidate = -1;
                    } else if (active.isEmpty() || !"auto".equals(deviceMode) ||
                               (detected == routeCandidate && now - candidateSince >= ROUTE_CONFIRM_MS)) {
                        device = detected;
                        routeCandidate = -1;
                    } else {
                        if (detected != routeCandidate) candidateSince = now;
                        routeCandidate = detected;
                        nextRefresh = candidateSince + ROUTE_CONFIRM_MS;
                    }
                }
                if (!watchingPlayback) {
                    nextRefresh = 0;
                    routeCandidate = -1;
                }

                deadline = watchingPlayback ? nextRefresh : Long.MAX_VALUE;
                if (!playbackCallbacks) deadline = Math.min(deadline, now + FALLBACK_IDLE_MS);

                // Keep a paused session attached briefly so resuming keeps its DSP state.
                Iterator<Map.Entry<Integer, SessionEffect>> it = active.entrySet().iterator();
                while (it.hasNext()) {
                    Map.Entry<Integer, SessionEffect> entry = it.next();
                    SessionEffect se = entry.getValue();
                    String pkg = sessions.get(entry.getKey());
                    if (se.pkg.equals(pkg)) {
                        se.releaseAt = 0;
                        setModelsPaused(se, false);
                        continue;
                    }
                    if (pkg == null && apps.containsKey(se.pkg)) {
                        if (se.releaseAt == 0) se.releaseAt = now + SESSION_GRACE_MS;
                        if (se.releaseAt > now) {
                            deadline = Math.min(deadline, se.releaseAt);
                            long pauseAt = se.releaseAt - SESSION_GRACE_MS + MODEL_PAUSE_MS;
                            if (now >= pauseAt) {
                                setModelsPaused(se, true);
                            } else {
                                deadline = Math.min(deadline, pauseAt);
                            }
                            continue;
                        }
                    }
                    releaseEffect(se.effect);
                    it.remove();
                }

                for (Map.Entry<Integer, String> session : sessions.entrySet()) {
                    int sid = session.getKey();
                    String pkg = session.getValue();
                    int prof = apps.get(pkg);

                    int sessReverb = (prof == 3) ? reverbMusic
                                   : (prof == 4 ? reverbPodcast : reverbMovie);
                    int sessScene = (prof == 3) ? sceneMusic
                                  : (prof == 2 ? sceneMovie : (prof == 4 ? scenePodcast : 0));

                    SessionEffect se = active.get(sid);
                    if (se == null) {
                        // Do not hammer AudioFlinger: one attempt per 15 s per session
                        // while the effect cannot be created.
                        Long failed = failedAt.get(sid);
                        if (failed != null && now - failed < SESSION_GRACE_MS) continue;
                        AudioEffect fx = null;
                        try {
                            fx = createAllyEffect(sid);
                            setAllyParam(fx, prof, neural, sessReverb, upmix, autoEq, sessScene, device);
                            int result = fx.setEnabled(true);
                            if (result != AudioEffect.SUCCESS) {
                                throw new IllegalStateException("AudioEffect.setEnabled returned " + result);
                            }
                            se = new SessionEffect();
                            se.effect = fx;
                            se.pkg = pkg;
                            se.profile = prof;
                            se.neural = neural;
                            se.reverb = sessReverb;
                            se.upmix = upmix;
                            se.autoEq = autoEq;
                            se.sceneAdapter = sessScene;
                            se.device = device;
                            active.put(sid, se);
                            failedAt.remove(sid);
                            lastError = null;
                        } catch (Exception e) {
                            if (fx != null) releaseEffect(fx);
                            failedAt.put(sid, now);
                            lastError = e.getClass().getSimpleName() + ": " + e.getMessage();
                        }
                    } else if (se.profile != prof || se.neural != neural || se.reverb != sessReverb ||
                               se.upmix != upmix || se.autoEq != autoEq || se.sceneAdapter != sessScene ||
                               se.device != device) {
                        try {
                            setAllyParam(se.effect, prof, neural, sessReverb, upmix, autoEq, sessScene, device);
                            se.profile = prof;
                            se.neural = neural;
                            se.reverb = sessReverb;
                            se.upmix = upmix;
                            se.autoEq = autoEq;
                            se.sceneAdapter = sessScene;
                            se.device = device;
                            lastError = null;
                        } catch (Exception e) {
                            lastError = e.getClass().getSimpleName() + ": " + e.getMessage();
                        }
                    }
                }
                lastStatus = writeStatus(device, lastError, lastStatus);
            } catch (InterruptedException ignored) {
                break;
            } catch (Exception e) {
                lastError = e.getClass().getSimpleName() + ": " + e.getMessage();
                lastStatus = writeStatus(device, lastError, lastStatus);
                deadline = SystemClock.elapsedRealtime() + ACTIVE_REFRESH_MS;
            }
        }
        observer.stopWatching();
        for (SessionEffect se : active.values()) releaseEffect(se.effect);
    }
}
