package newmtsa.synthesis;

import ai.onnxruntime.*;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

/**
 * Benchmark runner: evaluates a trained ONNX model against a directory of FSP instances.
 *
 * <pre>
 * Usage: DCSBenchmark &lt;onnx_path&gt; &lt;fsp_dir&gt; [budget=2500]
 * </pre>
 *
 * Model type (flat / lstm / transformer) is auto-detected from ONNX input names.
 * Pruning: if (N,1) is not solved within budget, all instances with N'&gt;=N are skipped.
 */
public class DCSBenchmark {

    private interface RLPolicy extends AutoCloseable {
        void resetEpisode();
        int  selectAction(List<int[]> frontier) throws OrtException;
        @Override void close() throws OrtException;
    }

    private static final class FlatPolicy implements RLPolicy {
        private final OrtEnvironment env;
        private final OrtSession     session;
        private final String         outputName;

        FlatPolicy(OrtEnvironment env, OrtSession session) throws OrtException {
            this.env        = env;
            this.session    = session;
            this.outputName = session.getOutputNames().iterator().next();
        }

        @Override public void resetEpisode() {}

        @Override
        public int selectAction(List<int[]> frontier) throws OrtException {
            int N = frontier.size(), F = frontier.get(0).length;
            float[][] data = new float[N][F];
            for (int i = 0; i < N; i++)
                for (int j = 0; j < F; j++) data[i][j] = frontier.get(i)[j];
            try (OnnxTensor t   = OnnxTensor.createTensor(env, data);
                 var        res = session.run(Map.of("features", t))) {
                return argmax((float[]) res.get(outputName).orElseThrow().getValue());
            }
        }

        @Override public void close() {}
    }

    private static final class LSTMPolicy implements RLPolicy {
        private final OrtEnvironment env;
        private final OrtSession     session;
        private final int            H;
        private float[][][]          prevFeat;
        private float[][][]          h;
        private float[][][]          c;

        LSTMPolicy(OrtEnvironment env, OrtSession session) throws OrtException {
            this.env     = env;
            this.session = session;
            TensorInfo hInfo = (TensorInfo) session.getOutputInfo().get("h_out").getInfo();
            this.H = (int) hInfo.getShape()[2];
        }

        @Override
        public void resetEpisode() {
            prevFeat = null;
            h = new float[1][1][H];
            c = new float[1][1][H];
        }

        @Override
        public int selectAction(List<int[]> frontier) throws OrtException {
            int N = frontier.size(), F = frontier.get(0).length;
            if (prevFeat == null) prevFeat = new float[1][1][F];

            float[][] cands = new float[N][F];
            for (int i = 0; i < N; i++)
                for (int j = 0; j < F; j++) cands[i][j] = frontier.get(i)[j];

            try (OnnxTensor tP = OnnxTensor.createTensor(env, prevFeat);
                 OnnxTensor tH = OnnxTensor.createTensor(env, h);
                 OnnxTensor tC = OnnxTensor.createTensor(env, c);
                 OnnxTensor tK = OnnxTensor.createTensor(env, cands)) {

                Map<String, OnnxTensor> inputs = new LinkedHashMap<>();
                inputs.put("prev_feat",  tP);
                inputs.put("h_in",       tH);
                inputs.put("c_in",       tC);
                inputs.put("candidates", tK);

                try (var res = session.run(inputs)) {
                    float[] scores = (float[]) res.get("scores").orElseThrow().getValue();
                    int action = argmax(scores);

                    h = (float[][][]) res.get("h_out").orElseThrow().getValue();
                    c = (float[][][]) res.get("c_out").orElseThrow().getValue();

                    prevFeat = new float[1][1][F];
                    int[] chosen = frontier.get(action);
                    for (int j = 0; j < F; j++) prevFeat[0][0][j] = chosen[j];
                    return action;
                }
            }
        }

        @Override public void close() {}
    }

    private static final class TransformerPolicy implements RLPolicy {
        private final OrtEnvironment env;
        private final OrtSession     session;
        private List<float[]>        history;
        private float[]              prevFeat;

        TransformerPolicy(OrtEnvironment env, OrtSession session) {
            this.env     = env;
            this.session = session;
        }

        @Override
        public void resetEpisode() {
            history  = new ArrayList<>();
            prevFeat = null;
        }

        @Override
        public int selectAction(List<int[]> frontier) throws OrtException {
            int N = frontier.size(), F = frontier.get(0).length;
            if (prevFeat == null) prevFeat = new float[F];

            history.add(prevFeat.clone());
            int T = history.size();
            float[][][] seq = new float[1][T][F];
            for (int t = 0; t < T; t++) seq[0][t] = history.get(t);

            float[][] cands = new float[N][F];
            for (int i = 0; i < N; i++)
                for (int j = 0; j < F; j++) cands[i][j] = frontier.get(i)[j];

            try (OnnxTensor tS = OnnxTensor.createTensor(env, seq);
                 OnnxTensor tK = OnnxTensor.createTensor(env, cands);
                 var        res = session.run(Map.of("sequence", tS, "candidates", tK))) {
                float[] scores = (float[]) res.get("scores").orElseThrow().getValue();
                int action = argmax(scores);

                prevFeat = new float[F];
                int[] chosen = frontier.get(action);
                for (int j = 0; j < F; j++) prevFeat[j] = chosen[j];
                return action;
            }
        }

        @Override public void close() {}
    }

    private record BenchResult(boolean solved, boolean realizable, int transitions, int states) {}

    private static String detectModelType(OrtSession session) throws OrtException {
        Set<String> ins = session.getInputNames();
        if (ins.contains("sequence"))  return "transformer";
        if (ins.contains("prev_feat")) return "lstm";
        return "flat";
    }

    /** Reads the fixed feature-vector size F from the ONNX model's input metadata. */
    private static int extractExpectedFeatureSize(OrtSession session,
                                                   String modelType) throws OrtException {
        String inputName = modelType.equals("flat") ? "features" : "candidates";
        NodeInfo info = session.getInputInfo().get(inputName);
        long[] shape  = ((TensorInfo) info.getInfo()).getShape();
        return (int) shape[shape.length - 1]; // last dim is always F
    }

    private static RLPolicy createPolicy(OrtEnvironment ortEnv, OrtSession session,
                                         String modelType) throws OrtException {
        return switch (modelType) {
            case "lstm"        -> new LSTMPolicy(ortEnv, session);
            case "transformer" -> new TransformerPolicy(ortEnv, session);
            default            -> new FlatPolicy(ortEnv, session);
        };
    }

    private static DCSForPython loadInstance(Path fspPath, String featureType) throws IOException {
        // Heuristic is required by the constructor but never called in RL mode —
        // the ONNX policy overrides all action selection. FIRST is the cheapest placeholder.
        return DCSForPython.fromPath(fspPath.toString(), "FIRST", featureType);
    }

    private static String extractAgentFromPath(String onnxPath) {
        for (Path part : Paths.get(onnxPath)) {
            String s = part.toString().toLowerCase();
            if (s.startsWith("dqn")) return "dqn";
            if (s.startsWith("ppo")) return "ppo";
            if (s.startsWith("sac")) return "sac";
        }
        return "rl";
    }

    private static String extractFeatureFromPath(String onnxPath) {
        for (Path part : Paths.get(onnxPath)) {
            String s = part.toString().toLowerCase();
            if (s.equals("basic")) return "basic";
            if (s.equals("rol"))   return "rol";
        }
        return DEFAULT_FEATURE_TYPE.toLowerCase();
    }

    private static String extractNetworkFromPath(String onnxPath) {
        for (Path part : Paths.get(onnxPath)) {
            String s = part.toString().toLowerCase();
            if (s.contains("transformer")) return "transformer";
            if (s.contains("lstm"))        return "lstm";
            if (s.contains("flat"))        return "flat";
        }
        return "flat";
    }

    private static BenchResult runEpisode(DCSForPython dcs, RLPolicy policy,
                                          int budget) throws OrtException {
        policy.resetEpisode();
        while (!dcs.isExplorationEnded() && dcs.getTransitionsExplored() < budget) {
            List<int[]> frontier = dcs.getFrontierWithFeatures();
            if (frontier.isEmpty()) break;
            dcs.expand(policy.selectAction(frontier));
        }
        boolean finished = dcs.isExplorationEnded();
        return new BenchResult(finished, finished && dcs.isRealizable(),
                               dcs.getTransitionsExplored(), dcs.getStatesExplored());
    }

    private static Path findProjectRoot(Path knownPath) {
        Path p = knownPath.toAbsolutePath().normalize();
        if (!Files.isDirectory(p)) p = p.getParent();
        while (p != null) {
            if (Files.exists(p.resolve("pom.xml"))) return p;
            p = p.getParent();
        }
        return Paths.get("").toAbsolutePath();
    }

    private static String deriveFamily(Path fsp) {
        String name = fsp.getFileName().toString().replace(".fsp", "");
        int last = name.lastIndexOf('-');
        int prev = last > 0 ? name.lastIndexOf('-', last - 1) : -1;
        return prev > 0 ? name.substring(0, prev) : name;
    }

    private static int argmax(float[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
        return best;
    }

    /**
     * Benchmark a trained ONNX model against all FSP instances in a directory.
     *
     * <pre>
     * Usage: DCSBenchmark &lt;onnx_path&gt; &lt;fsp_dir&gt; [budget=2500]
     * </pre>
     */
    // ── heuristic mode ───────────────────────────────────────────────────────

    private static BenchResult runHeuristicEpisode(DCSForPython dcs, int budget) {
        while (!dcs.isExplorationEnded() && dcs.getTransitionsExplored() < budget) {
            List<ExtendedTransition> frontier = dcs.getFrontier();
            if (frontier.isEmpty()) break;
            dcs.expand(dcs.heuristic.pick(frontier));
        }
        boolean finished = dcs.isExplorationEnded();
        return new BenchResult(finished, finished && dcs.isRealizable(),
                               dcs.getTransitionsExplored(), dcs.getStatesExplored());
    }

    private static void runHeuristicBenchmark(Path fspDir, String heuristicType,
                                               int budget, String csvPath,
                                               boolean saveCsv) throws Exception {
        System.out.printf("Heuristic : %s%nBudget    : %d transitions%n%n", heuristicType, budget);

        Pattern pat = Pattern.compile(".+-(\\d+)-(\\d+)\\.fsp");
        Map<Integer, List<Map.Entry<Integer, Path>>> byN = new TreeMap<>();

        try (var stream = Files.walk(fspDir)) {
            stream.filter(p -> p.toString().endsWith(".fsp"))
                  .forEach(p -> {
                      Matcher m = pat.matcher(p.getFileName().toString());
                      if (m.matches()) {
                          int n = Integer.parseInt(m.group(1));
                          int k = Integer.parseInt(m.group(2));
                          byN.computeIfAbsent(n, x -> new ArrayList<>())
                             .add(Map.entry(k, p));
                      }
                  });
        }
        byN.values().forEach(list -> list.sort(Map.Entry.comparingByKey()));

        if (saveCsv) {
            Path parent = Paths.get(csvPath).getParent();
            if (parent != null) Files.createDirectories(parent);
        }

        boolean writeHeader = saveCsv &&
            (!Files.exists(Paths.get(csvPath)) || Files.size(Paths.get(csvPath)) == 0);
        try (PrintWriter csv = saveCsv
                ? new PrintWriter(new FileWriter(csvPath, true))
                : new PrintWriter(OutputStream.nullOutputStream())) {
            if (writeHeader) {
                csv.println("family,n,k,solved,realizable,transitions,states,time_ms");
                csv.flush();
            }

            int skipFromN = Integer.MAX_VALUE;

            for (var nEntry : byN.entrySet()) {
                int     n         = nEntry.getKey();
                boolean skipRestK = false;

                for (var kEntry : nEntry.getValue()) {
                    int    k      = kEntry.getKey();
                    Path   fsp    = kEntry.getValue();
                    String family = deriveFamily(fsp);

                    if (n >= skipFromN || skipRestK) {
                        System.out.printf("  N=%-3d K=%-3d  SKIPPED (pruned)%n", n, k);
                        csv.printf("%s,%d,%d,false,???,-1,-1,-1%n", family, n, k);
                        csv.flush();
                        continue;
                    }

                    DCSForPython dcs;
                    try {
                        dcs = DCSForPython.fromPath(fsp.toString(), heuristicType, "");
                    } catch (Exception e) {
                        System.err.printf("  N=%-3d K=%-3d  ERROR loading: %s%n",
                            n, k, e.getMessage());
                        continue;
                    }

                    long        t0          = System.currentTimeMillis();
                    BenchResult res         = runHeuristicEpisode(dcs, budget);
                    long        ms          = System.currentTimeMillis() - t0;
                    String      realizable  = res.solved() ? String.valueOf(res.realizable()) : "???";

                    System.out.printf(
                        "  N=%-3d K=%-3d  solved=%-5s realizable=%-5s trans=%-5d states=%-5d %dms%n",
                        n, k, res.solved(), realizable,
                        res.transitions(), res.states(), ms);

                    csv.printf("%s,%d,%d,%b,%s,%d,%d,%d%n",
                        family, n, k,
                        res.solved(), realizable,
                        res.transitions(), res.states(), ms);
                    csv.flush();

                    if (!res.solved()) {
                        skipRestK = true;
                        if (k == 1) skipFromN = n;
                    }
                }
            }
        }
        if (saveCsv) System.out.println("\nDone. Results written to: " + csvPath);
        else         System.out.println("\nDone. (CSV save disabled)");
    }

    // ── hardcoded defaults (used when no CLI args are provided) ──────────────
    // Heuristic mode default. "RL" = RL model mode (uses DEFAULT_ONNX_PATH).
    // Options: RL, FIRST, RANDOM, BFS, RA, RA_R, RA_E, RA_ER, RA_ERG,
    //          RA_OPEN, RA_R_OPEN, RA_E_OPEN, RA_ER_OPEN, RA_ERG_OPEN
    private static final String  DEFAULT_HEURISTIC_TYPE = "RL";
    // Feature set used when loading RL instances. Options: BASIC, ROL
    private static final String  DEFAULT_FEATURE_TYPE   = "ROL";
    private static final String  DEFAULT_ONNX_PATH     = ".\\python\\results\\blocking\\TL\\rol\\ppo_flat\\ppo_ep0725.onnx";
    private static final String  DEFAULT_FSP_DIR        = ".\\fsp\\Blocking\\Benchmark\\TL\\";
    private static final int     DEFAULT_BUDGET          = 2500;
    private static final boolean DEFAULT_SAVE_CSV        = true;

    /**
     * Usage:
     *   Heuristic mode : DCSBenchmark <HEURISTIC_TYPE> <fsp_dir> [budget=2500]
     *   RL model mode  : DCSBenchmark <model.onnx>    <fsp_dir> [budget=2500]
     *   No args        : uses hardcoded defaults above
     * Mode is auto-detected: if args[0] ends with ".onnx" → RL mode, otherwise → heuristic mode. */
    public static void main(String[] args) throws Exception {
        Path   fspDir;
        int    budget;
        String heuristicType;
        String onnxPath;

        if (args.length == 0) {
            fspDir        = Paths.get(DEFAULT_FSP_DIR);
            budget        = DEFAULT_BUDGET;
            heuristicType = DEFAULT_HEURISTIC_TYPE;
            onnxPath      = DEFAULT_ONNX_PATH;
            System.out.println("[DCSBenchmark] No args — using hardcoded defaults:");
            if (!"RL".equals(heuristicType)) {
                System.out.printf("  mode  = heuristic (%s)%n  dir   = %s%n  budget= %d%n%n",
                    heuristicType, fspDir, budget);
            } else {
                System.out.printf("  onnx  = %s%n  dir   = %s%n  budget= %d%n%n",
                    onnxPath, fspDir, budget);
            }
        } else if (args.length < 2) {
            System.err.println(
                "Usage (heuristic): DCSBenchmark <HEURISTIC_TYPE> <fsp_dir> [budget=2500]");
            System.err.println(
                "Usage (RL model) : DCSBenchmark <model.onnx> <fsp_dir> [budget=2500]");
            System.exit(1);
            return;
        } else {
            fspDir        = Paths.get(args[1]);
            budget        = args.length > 2 ? Integer.parseInt(args[2]) : DEFAULT_BUDGET;
            heuristicType = args[0].endsWith(".onnx") ? "RL" : args[0].toUpperCase();
            onnxPath      = args[0].endsWith(".onnx") ? args[0] : null;
        }

        if (!"RL".equals(heuristicType)) {
            Path   root     = findProjectRoot(fspDir);
            String csvPath  = root.resolve("python").resolve("results")
                                  .resolve(heuristicType.toLowerCase() + "_benchmark.csv")
                                  .toString();
            runHeuristicBenchmark(fspDir, heuristicType, budget, csvPath, DEFAULT_SAVE_CSV);
            return;
        }

        String agentType   = extractAgentFromPath(onnxPath);
        String featureType = extractFeatureFromPath(onnxPath);
        String networkType = extractNetworkFromPath(onnxPath);
        Path   root        = findProjectRoot(Paths.get(onnxPath));
        String csvPath     = root.resolve("python").resolve("results")
                                 .resolve(agentType + "_" + networkType + "_" + featureType + "_benchmark.csv")
                                 .toString();

        OrtEnvironment ortEnv = OrtEnvironment.getEnvironment();
        try (OrtSession session = ortEnv.createSession(onnxPath)) {
            String modelType  = detectModelType(session);
            int    expectedF  = extractExpectedFeatureSize(session, modelType);
            System.out.printf("Model : %s%nType  : %s%nFeature: %s%nF     : %d features%nBudget: %d transitions%n%n",
                Paths.get(onnxPath).getFileName(), modelType, featureType.toUpperCase(), expectedF, budget);

            Pattern pat = Pattern.compile(".+-(\\d+)-(\\d+)\\.fsp");
            Map<Integer, List<Map.Entry<Integer, Path>>> byN = new TreeMap<>();

            try (var stream = Files.walk(fspDir)) {
                stream.filter(p -> p.toString().endsWith(".fsp"))
                      .forEach(p -> {
                          Matcher m = pat.matcher(p.getFileName().toString());
                          if (m.matches()) {
                              int n = Integer.parseInt(m.group(1));
                              int k = Integer.parseInt(m.group(2));
                              byN.computeIfAbsent(n, x -> new ArrayList<>())
                                 .add(Map.entry(k, p));
                          }
                      });
            }
            byN.values().forEach(list -> list.sort(Map.Entry.comparingByKey()));

            boolean writeHeader = DEFAULT_SAVE_CSV &&
                (!Files.exists(Paths.get(csvPath)) || Files.size(Paths.get(csvPath)) == 0);
            try (PrintWriter csv = DEFAULT_SAVE_CSV
                    ? new PrintWriter(new FileWriter(csvPath, true))
                    : new PrintWriter(OutputStream.nullOutputStream())) {
                if (writeHeader) {
                    csv.println("family,n,k,solved,realizable,transitions,states,time_ms,agent_path");
                    csv.flush();
                }

                int skipFromN = Integer.MAX_VALUE;

                for (var nEntry : byN.entrySet()) {
                    int     n         = nEntry.getKey();
                    boolean skipRestK = false;

                    for (var kEntry : nEntry.getValue()) {
                        int    k      = kEntry.getKey();
                        Path   fsp    = kEntry.getValue();
                        String family = deriveFamily(fsp);

                        if (n >= skipFromN || skipRestK) {
                            System.out.printf("  N=%-3d K=%-3d  SKIPPED (pruned)%n", n, k);
                            csv.printf("%s,%d,%d,false,???,-1,-1,-1,%s%n", family, n, k, onnxPath);
                            csv.flush();
                            continue;
                        }

                        DCSForPython dcs;
                        try {
                            dcs = loadInstance(fsp, featureType.toUpperCase());
                        } catch (Exception e) {
                            System.err.printf("  N=%-3d K=%-3d  ERROR loading: %s%n",
                                n, k, e.getMessage());
                            continue;
                        }

                        if (!dcs.isExplorationEnded()) {
                            List<int[]> probe = dcs.getFrontierWithFeatures();
                            if (!probe.isEmpty() && probe.get(0).length != expectedF) {
                                System.err.printf(
                                    "  N=%-3d K=%-3d  SKIP: feature size mismatch" +
                                    " (model expects %d, instance has %d)%n",
                                    n, k, expectedF, probe.get(0).length);
                                csv.printf("%s,%d,%d,false,???,-1,-1,-1,%s%n", family, n, k, onnxPath);
                                csv.flush();
                                continue;
                            }
                        }

                        try (RLPolicy policy = createPolicy(ortEnv, session, modelType)) {
                            long        t0         = System.currentTimeMillis();
                            BenchResult res        = runEpisode(dcs, policy, budget);
                            long        ms         = System.currentTimeMillis() - t0;
                            String      realizable = res.solved() ? String.valueOf(res.realizable()) : "???";

                            System.out.printf(
                                "  N=%-3d K=%-3d  solved=%-5s realizable=%-5s trans=%-5d %dms%n",
                                n, k, res.solved(), realizable, res.transitions(), ms);

                            csv.printf("%s,%d,%d,%b,%s,%d,%d,%d,%s%n",
                                family, n, k,
                                res.solved(), realizable,
                                res.transitions(), res.states(), ms, onnxPath);
                            csv.flush();

                            if (!res.solved()) {
                                skipRestK = true;
                                if (k == 1) skipFromN = n;
                            }
                        }
                    }
                }
            }
        }
        if (DEFAULT_SAVE_CSV) System.out.println("\nDone. Results written to: " + csvPath);
        else                  System.out.println("\nDone. (CSV save disabled)");
    }
}
