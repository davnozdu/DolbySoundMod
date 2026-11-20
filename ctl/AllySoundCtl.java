import android.media.audiofx.AudioEffect;
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
    private static final String STATUS_JSON = BASE_DIR + "/status.json";

    private static class SessionEffect {
        AudioEffect effect;
        String pkg;
        int profile;
        int neural;
        int reverb;
        int device;
    }

    private static final Map<Integer, SessionEffect> active = new HashMap<>();
    private static final Map<Integer, Long> failedAt = new HashMap<>();

    private static int parseProfile(String s) {
        if (s == null) return 0;
        switch (s.trim().toLowerCase()) {
            case "game": return 1;
            case "movie": return 2;
            case "music": return 3;
            default: return 0;
        }
    }

    private static String profileName(int p) {
        switch (p) {
            case 1: return "game";
            case 2: return "movie";
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

    private static int detectOutputDevice(String mode) {
        if (!"auto".equals(mode)) return parseDevice(mode);
        try {
            Process p = Runtime.getRuntime().exec(new String[]{"dumpsys", "audio"});
            BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String line;
            boolean inMedia = false;
            int detected = 1; // default speakers
            while ((line = br.readLine()) != null) {
                String low = line.toLowerCase();
                if (low.contains("stream_music") || low.contains("devices for strategy")) {
                    inMedia = true;
                }
                if (inMedia) {
                    if (low.contains("xreal") || low.contains("nreal") || low.contains("viture") || low.contains("rokid")) {
                        detected = 2; // glasses
                        break;
                    }
                    if (low.contains("a2dp") || low.contains("ble_headset") || low.contains("wired_head") ||
                        low.contains("usb_headset") || low.contains("hearing_aid")) {
                        detected = 0; // headphones
                        break;
                    }
                    if (low.contains("speaker")) {
                        detected = 1; // speakers
                    }
                }
            }
            p.destroy();  // never waitFor() after an early break: dumpsys keeps
            return detected;  // writing into the pipe and waitFor() deadlocks
        } catch (Exception e) {
            return 0;
        }
    }

    private static Map<Integer, String> readActiveSessions() {
        Map<Integer, String> sessions = new HashMap<>();
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
                            sessions.put(sid, pkg);
                        } catch (NumberFormatException ignored) {}
                    }
                }
            }
            p.destroy();  // an early break leaves dumpsys blocked on a full pipe;
                          // waitFor() would then hang the whole control loop
        } catch (Exception ignored) {}
        return sessions;
    }

    private static AudioEffect createAllyEffect(int sessionId) throws Exception {
        Constructor<AudioEffect> ctor = AudioEffect.class.getDeclaredConstructor(
            UUID.class, UUID.class, int.class, int.class);
        ctor.setAccessible(true);
        return ctor.newInstance(TYPE_LOUDNESS, IMPL_ALLY, 0, sessionId);
    }

    private static void setAllyParam(AudioEffect fx, int profile, int neural, int reverb, int device) throws Exception {
        Method setParam = AudioEffect.class.getDeclaredMethod("setParameter", int.class, int.class);
        setParam.setAccessible(true);
        int code = 20000 + profile * 1000 + neural * 100 + reverb * 10 + device;
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
        while (true) {
            try {
                Map<String, Integer> apps = readAppsConf();
                String devMode = readDeviceMode();
                int device = detectOutputDevice(devMode);
                int neural = readNeuralEnabled();
                int reverb = readReverbEnabled();
                Map<Integer, String> sessions = readActiveSessions();

                // Remove stale or disabled sessions
                Iterator<Map.Entry<Integer, SessionEffect>> it = active.entrySet().iterator();
                while (it.hasNext()) {
                    Map.Entry<Integer, SessionEffect> entry = it.next();
                    int sid = entry.getKey();
                    SessionEffect se = entry.getValue();
                    String pkg = sessions.get(sid);
                    Integer wantedProf = (pkg != null) ? apps.get(pkg) : null;
                    if (pkg == null || wantedProf == null || wantedProf == 0) {
                        try {
                            se.effect.setEnabled(false);
                            se.effect.release();
                        } catch (Exception ignored) {}
                        it.remove();
                    }
                }

                // Attach or update matching sessions
                Set<Integer> keys = new HashSet<>(sessions.keySet());
                for (int sid : keys) {
                    String pkg = sessions.get(sid);
                    Integer prof = apps.get(pkg);
                    if (prof == null || prof == 0) continue;

                    SessionEffect se = active.get(sid);
                    if (se == null) {
                        // Do not hammer AudioFlinger: one attempt per 15 s per session
                        // while the effect cannot be created.
                        Long failed = failedAt.get(sid);
                        if (failed != null && System.currentTimeMillis() - failed < 15000) continue;
                        try {
                            AudioEffect fx = createAllyEffect(sid);
                            setAllyParam(fx, prof, neural, reverb, device);
                            fx.setEnabled(true);
                            se = new SessionEffect();
                            se.effect = fx;
                            se.pkg = pkg;
                            se.profile = prof;
                            se.neural = neural;
                            se.reverb = reverb;
                            se.device = device;
                            active.put(sid, se);
                            failedAt.remove(sid);
                            lastError = null;
                        } catch (Exception e) {
                            failedAt.put(sid, System.currentTimeMillis());
                            lastError = e.getClass().getSimpleName() + ": " + e.getMessage();
                        }
                    } else if (se.profile != prof || se.neural != neural || se.reverb != reverb ||
                               se.device != device) {
                        try {
                            setAllyParam(se.effect, prof, neural, reverb, device);
                            se.profile = prof;
                            se.neural = neural;
                            se.reverb = reverb;
                            se.device = device;
                            lastError = null;
                        } catch (Exception e) {
                            lastError = e.getClass().getSimpleName() + ": " + e.getMessage();
                        }
                    }
                }
                writeStatus(device, lastError);
            } catch (Exception e) {
                lastError = e.getMessage();
                writeStatus(0, lastError);
            }

            try {
                Thread.sleep(1500);
            } catch (InterruptedException ignored) {
                break;
            }
        }
    }
}
