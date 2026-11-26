import android.media.audiofx.AudioEffect;
import android.os.IBinder;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.nio.channels.FileLock;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
    }

    private static final Map<Integer, SessionEffect> active = new HashMap<>();
    private static final Map<Integer, Long> failedAt = new HashMap<>();
    private static final Map<Integer, Long> lastSeen = new HashMap<>();
    private static final long SESSION_GRACE_MS = 15000;

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

    private static int readNeuralEnabled() {
        File f = new File(NEURAL_CONF);
        if (!f.exists()) return 0;
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String line = br.readLine();
            return (line != null && "1".equals(line.trim())) ? 1 : 0;
        } catch (Exception e) {
            return 0;
        }
    }

    private static int readReverbEnabled() {
        File f = new File(REVERB_CONF);
        if (!f.exists()) return 0;
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String line = br.readLine();
            return (line != null && "1".equals(line.trim())) ? 1 : 0;
        } catch (Exception e) {
            return 0;
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

    private static int detectOutputDevice(String mode) {
        if (!"auto".equals(mode)) return parseDevice(mode);
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
                if (low.contains("xreal") || low.contains("nreal") || low.contains("viture") || low.contains("rokid")) {
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
            return 1;
        } finally {
            if (p != null) p.destroy();
        }
        if (route == null || route.isEmpty()) return 1; // unknown: phone speaker is the safe guess
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
        try {
            Process p = Runtime.getRuntime().exec(new String[]{"dumpsys", "audio"});
            BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String line;
            while ((line = br.readLine()) != null) {
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
        if (!sessions.isEmpty()) {
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

    // Fast path: ask the audio service for its playback configurations over binder.
    // One call costs well under a millisecond, against 40-90 ms for `dumpsys audio`,
    // so new playbacks can be picked up every 250 ms and the effect attaches within
    // a fraction of a second of the track starting.
    private static Object audioService;
    private static Method getPlaybackConfigs;
    private static boolean fastSessionsBroken;
    private static final int PLAYER_STATE_STARTED = 2;

    private static Map<Integer, String> readSessionsFast() throws Exception {
        if (getPlaybackConfigs == null) {
            Class<?> serviceManager = Class.forName("android.os.ServiceManager");
            IBinder binder = (IBinder) serviceManager.getMethod("getService", String.class).invoke(null, "audio");
            Class<?> stub = Class.forName("android.media.IAudioService$Stub");
            audioService = stub.getMethod("asInterface", IBinder.class).invoke(null, binder);
            getPlaybackConfigs = audioService.getClass().getMethod("getActivePlaybackConfigurations");
        }
        Map<Integer, String> sessions = new HashMap<>();
        List<?> configs = (List<?>) getPlaybackConfigs.invoke(audioService);
        if (configs == null) return sessions;
        for (Object config : configs) {
            Class<?> type = config.getClass();
            int state = (Integer) type.getMethod("getPlayerState").invoke(config);
            if (state != PLAYER_STATE_STARTED) continue;
            int sid = (Integer) type.getMethod("getSessionId").invoke(config);
            int pid = (Integer) type.getMethod("getClientPid").invoke(config);
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
        Method setParam = AudioEffect.class.getDeclaredMethod("setParameter", int.class, int.class);
        setParam.setAccessible(true);
        int flags = sceneAdapter * 4 + autoEq * 2 + upmix;
        int code = 50000 + flags * 10000 + profile * 1000 + neural * 100 + reverb * 10 + device;
        setParam.invoke(fx, 0 /* PARAM_TARGET_GAIN_MB */, code);
    }

    private static void writeStatus(int device, String error) {
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
              .append(",\"neural\":").append(se.neural != 0).append("}");
        }
        sb.append("]}");
        try (FileWriter fw = new FileWriter(STATUS_JSON)) {
            fw.write(sb.toString());
        } catch (Exception ignored) {}
    }

    public static void main(String[] args) {
        new File(BASE_DIR).mkdirs();
        // Single instance only: the service.sh supervisor restarts us every 3s,
        // and two controllers would attach two effects to the same session.
        RandomAccessFile lockFile = null;
        FileLock lock = null;
        try {
            lockFile = new RandomAccessFile(BASE_DIR + "/ctl.lock", "rw");
            lock = lockFile.getChannel().tryLock();
            if (lock == null) {
                return;
            }
        } catch (Exception ignored) {
            // lock is best effort; never block the controller
        }
        String lastError = null;
        // Sessions are polled every 250 ms over binder; settings, the output device
        // (`dumpsys audio`) and the status file are refreshed every 1.5 s.
        long tick = 0;
        Map<String, Integer> apps = null;
        int device = 0, neural = 0, upmix = 0, autoEq = 0;
        int reverbMusic = 1, reverbMovie = 0, reverbPodcast = 0;
        int sceneMusic = 1, sceneMovie = 0, scenePodcast = 0;
        while (true) {
            try {
                final boolean refresh = tick++ % 6 == 0 || apps == null;
                if (refresh) {
                    apps = readAppsConf();
                    device = detectOutputDevice(readDeviceMode());
                    neural = readNeuralEnabled();
                    int legacyReverb = readReverbEnabled();
                    reverbMusic = readToggleDefault(REVERB_MUSIC_CONF, 1);
                    reverbMovie = readToggleDefault(REVERB_MOVIE_CONF, legacyReverb);
                    reverbPodcast = readToggleDefault(REVERB_PODCAST_CONF, 0);
                    sceneMusic = readToggleDefault(SCENE_MUSIC_CONF, 1);
                    sceneMovie = readToggleDefault(SCENE_MOVIE_CONF, 0);
                    scenePodcast = readToggleDefault(SCENE_PODCAST_CONF, 0);
                    upmix = readToggle(UPMIX_CONF);
                    autoEq = readToggle(AUTO_EQ_CONF);
                }
                Map<Integer, String> sessions = null;
                if (!fastSessionsBroken) {
                    try {
                        sessions = readSessionsFast();
                    } catch (Throwable t) {
                        fastSessionsBroken = true;   // fall back to dumpsys for good
                    }
                }
                if (sessions == null) {
                    if (!refresh) {
                        Thread.sleep(250);
                        continue;
                    }
                    sessions = readActiveSessions();
                }

                // Remove stale or disabled sessions. A session that merely stopped
                // playing (pause, the gap between two tracks) keeps its effect for
                // SESSION_GRACE_MS, so resuming or the next track does not attach a
                // new effect and start it from scratch.
                final long now = System.currentTimeMillis();
                for (int sid : sessions.keySet()) lastSeen.put(sid, now);
                Iterator<Map.Entry<Integer, SessionEffect>> it = active.entrySet().iterator();
                while (it.hasNext()) {
                    Map.Entry<Integer, SessionEffect> entry = it.next();
                    int sid = entry.getKey();
                    SessionEffect se = entry.getValue();
                    String pkg = sessions.get(sid);
                    if (pkg == null) {
                        Long seen = lastSeen.get(sid);
                        if (seen != null && now - seen < SESSION_GRACE_MS && apps.get(se.pkg) != null) continue;
                    }
                    Integer wantedProf = (pkg != null) ? apps.get(pkg) : null;
                    if (pkg == null || wantedProf == null || wantedProf == 0) {
                        lastSeen.remove(sid);
                        try {
                            se.effect.setEnabled(false);
                            se.effect.release();
                        } catch (Exception ignored) {}
                        it.remove();
                    }
                }

                Iterator<Map.Entry<Integer, Long>> seenIt = lastSeen.entrySet().iterator();
                while (seenIt.hasNext()) {
                    Map.Entry<Integer, Long> seen = seenIt.next();
                    if (now - seen.getValue() > SESSION_GRACE_MS && !active.containsKey(seen.getKey())) seenIt.remove();
                }

                // Attach or update matching sessions
                Set<Integer> keys = new HashSet<>(sessions.keySet());
                for (int sid : keys) {
                    String pkg = sessions.get(sid);
                    Integer prof = apps.get(pkg);
                    if (prof == null || prof == 0) continue;

                    int sessReverb = (prof == 3) ? reverbMusic
                                   : (prof == 4 ? reverbPodcast : reverbMovie);
                    int sessScene = (prof == 3) ? sceneMusic
                                  : (prof == 2 ? sceneMovie : (prof == 4 ? scenePodcast : 0));

                    SessionEffect se = active.get(sid);
                    if (se == null) {
                        // Do not hammer AudioFlinger: one attempt per 15 s per session
                        // while the effect cannot be created.
                        Long failed = failedAt.get(sid);
                        if (failed != null && System.currentTimeMillis() - failed < 15000) continue;
                        try {
                            AudioEffect fx = createAllyEffect(sid);
                            setAllyParam(fx, prof, neural, sessReverb, upmix, autoEq, sessScene, device);
                            fx.setEnabled(true);
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
                            failedAt.put(sid, System.currentTimeMillis());
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
                if (refresh) writeStatus(device, lastError);
            } catch (Exception e) {
                lastError = e.getMessage();
                writeStatus(0, lastError);
            }

            try {
                Thread.sleep(250);
            } catch (InterruptedException ignored) {
                break;
            }
        }
    }
}
