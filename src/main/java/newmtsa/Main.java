package newmtsa;

import newmtsa.parser.FSPParser;
import newmtsa.parser.ast.*;
import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.Director;
import newmtsa.synthesis.OTFDirectedControlledSynthesis;
import newmtsa.synthesis.gr1.OTFDirectedControledSyntesisGR1;
import newmtsa.synthesis.features.FeatureCompute;
import newmtsa.synthesis.features.FeatureType;
import newmtsa.synthesis.heuristics.Heuristic;
import newmtsa.synthesis.heuristics.HeuristicType;
import newmtsa.synthesis.heuristics.HandHeuristic;
import newmtsa.synthesis.heuristics.RandomHeuristic;
import newmtsa.synthesis.heuristics.SynthesisContext;
import newmtsa.synthesis.nonblocking.OTFDirectedControledSyntesisNonBlocking;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class Main {

    // ── Heuristic selection ───────────────────────────────────────────────────
    // Change HEURISTIC to switch strategy. For Non-Blocking DCS, RANDOM also
    // uses SCRIPT (see below); all other heuristics ignore SCRIPT.
    //
    //   FIRST         – always picks the first transition in the frontier (deterministic)
    //   RANDOM        – uniform random; uses SCRIPT indices first, then falls back to random
    //   BFS           – breadth-first layer-by-layer expansion
    //   HUMAN         – interactive: prints the frontier and asks the user to choose
    //   RA            – Ready Abstraction (base, Pazos 2024)
    //   RA_R          – RA + recompute estimates when new marked states are discovered
    //   RA_E          – RA + structure-aware tie-breaking
    //   RA_ER         – RA.R + RA.E combined
    //   RA_ERG        – RA.R + RA.E + Goals-as-targets (all improvements)
    //
    //   Open-queue variants (DFS bias — restrict picks to states with no pending uncontrollables):
    //   RA_OPEN       – RA  + open queue  (best open-queue variant per Pazos §7.2)
    //   RA_R_OPEN     – RA.R + open queue
    //   RA_E_OPEN     – RA.E + open queue  (NOTE: combining open queue + RA.E is counterproductive)
    //   RA_ER_OPEN    – RA.ER + open queue
    //   RA_ERG_OPEN   – RA.ERG + open queue
    //
    //   RL            – Marker for Python-driven RL training; selection comes from a Python
    //                   agent (DQN/PPO/SAC) via DCSForPython.expand(idx) using features
    //                   BASIC or ROL. In a standalone Main run it falls back to FIRST.
    //   MCTS_RL       – MCTS guided by ONNX policy/value model (sys props mcts.onnx.path,
    //                   mcts.simulations, mcts.cpuct, mcts.depth).
    //static final HeuristicType HEURISTIC = HeuristicType.RA_ERG;
    //static final HeuristicType HEURISTIC = HeuristicType.RANDOM;
    //static final HeuristicType HEURISTIC = HeuristicType.FIRST;
    //static final HeuristicType HEURISTIC = HeuristicType.BFS;
    //static final HeuristicType HEURISTIC = HeuristicType.HUMAN;
    //static final HeuristicType HEURISTIC = HeuristicType.RA_R;
    //static final HeuristicType HEURISTIC = HeuristicType.RA_E;
    //static final HeuristicType HEURISTIC = HeuristicType.RA_ER;
    //static final HeuristicType HEURISTIC = HeuristicType.RA_ERG;
    //static final HeuristicType HEURISTIC = HeuristicType.RA_ERG_OPEN;
    //static final HeuristicType HEURISTIC = HeuristicType.HAND;
    static final HeuristicType HEURISTIC = HeuristicType.SUPER_DFS;
    //static final HeuristicType HEURISTIC = HeuristicType.RL;
    //static final HeuristicType HEURISTIC = HeuristicType.MCTS_RL;

    // Print composite states as names (false) or numeric indices (true).
    static final boolean USE_NUMERIC_STATE_IDS = false;

    static final boolean VERBOSE = true; // Print heuristic frontier and transition scores before each expansion (RA shows distances).
    static final boolean SAVE_SOL = true; // Flag to save the trace in a .sol file
    static final boolean FRONTIER_RESTRICTION = false; // When true: uncontrollable + single-controllable transitions are expanded immediately on state discovery; multiple-controllable transitions go to the frontier for heuristic selection.
    static final boolean SAVE_FEATURE_VECTORS = false; // When true: collect all feature vectors seen across runs and write them to a .txt file alongside the FSP.
    static final int     RANDOM_RUNS          = 10;    // Number of runs when HEURISTIC=RANDOM and SAVE_FEATURE_VECTORS=true; other heuristics always do 1 run.

    // Path of the ONNX model loaded by RL and MCTS_RL heuristics.
    static final String MODEL_PATH = ".\\python\\results\\blocking\\TL\\rol\\ppo_flat\\ppo_ep0710.onnx";

    // Possible features types:
    //  "BASIC"
    //  "ROL"
    static final String FEATURE_TYPE = "ROL";

    // Scripted frontier indices used when HEURISTIC = RANDOM.
    // At step k picks pending.get(SCRIPT[k]); falls back to random once exhausted.
    // List.of() = pure random from the start.
    static final List<Integer> SCRIPT = List.of(0, 3, 6, 9, 12, 14, 17, 18, 14, 15, 22, 25, 27, 22, 28, 17);   // unused when HEURISTIC != RANDOM (Expansion to solve TL-2-2 in the same way the original RA should.)
    //static final List<Integer> SCRIPT = List.of();

    // ── trace recording ───────────────────────────────────────────────────────

    record TraceStep(List<ExtendedTransition> frontier, int index, ExtendedTransition selected) {}

    static class TraceRecordingHeuristic implements Heuristic {
        private final Heuristic delegate;
        private final List<TraceStep> trace = new ArrayList<>();

        TraceRecordingHeuristic(Heuristic delegate) { this.delegate = delegate; }

        @Override
        public int pick(List<ExtendedTransition> pending) {
            int idx = delegate.pick(pending);
            int safe = Math.max(0, Math.min(idx, pending.size() - 1));
            trace.add(new TraceStep(new ArrayList<>(pending), safe, pending.get(safe)));
            return idx;
        }

        @Override public void init(SynthesisContext ctx) { delegate.init(ctx); }
        @Override public void printFrontier(List<ExtendedTransition> pending, int pickedIndex) {
            delegate.printFrontier(pending, pickedIndex);
        }
        @Override public void notifyExplorationEnd(newmtsa.synthesis.Director result) { delegate.notifyExplorationEnd(result); }

        List<TraceStep> getTrace() { return trace; }
    }

    // ── main ──────────────────────────────────────────────────────────────────

    public static void main(String[] args) throws IOException {
        Path file;
        if(args.length > 0){
            file = Path.of(args[0]);
        }else{
            //file = Path.of(".\\fsp\\Blocking\\Benchmark\\TL\\TL-15-15.fsp");
            //file = Path.of(".\\fsp\\Blocking\\Benchmark\\CM\\CM-3-2.fsp");
            //file = Path.of(".\\fsp\\Blocking\\Benchmark\\DP\\DP-11-15.fsp");
            //file = Path.of(".\\fsp\\Blocking\\Benchmark\\DP\\DP-2-2.fsp");
            //file = Path.of(".\\fsp\\Blocking\\Benchmark\\TA\\TA-2-2.fsp");
            //file = Path.of(".\\fsp\\Blocking\\Benchmark\\DP\\DP-4-4.fsp");
            file = Path.of(".\\fsp\\Blocking\\Benchmark\\AT\\AT-4-4.fsp");
            //file = Path.of(".\\fsp\\Blocking\\Benchmark\\BW\\BW-4-2.fsp");
            //file = Path.of(".\\fsp\\Blocking\\ControllableFSPs\\test21.lts");
            //file = Path.of(".\\fsp\\Blocking\\ControllableFSPs\\GR1Test43.lts");
            //file = Path.of("C:\\Users\\diort\\Downloads\\data\\krka_et_al_FSE14\\reference_models\\ElemNumber$NumberFormatStringTokenizer.lts");

        }

        FSPModel model = FSPParser.parse(file);

        System.out.println("===============================================");
        System.out.println("  File: " + file.getFileName());
        System.out.println("===============================================");

        FSPParser.printModel(model);

        // Make MODEL_PATH visible to RL and MCTS_RL via their existing sys-property lookups.
        // No-op for other heuristics.
        System.setProperty("rl.onnx.path",   MODEL_PATH);
        System.setProperty("mcts.onnx.path", MODEL_PATH);
        System.setProperty("feature_type", FEATURE_TYPE);
        System.setProperty("random.script", SCRIPT.toString());

        runSynthesis(model, true, file);

        if (SAVE_FEATURE_VECTORS) {
            try {
                saveFeatureVectors(model, file);
            } catch (Exception e) {
                System.err.println("WARNING: could not save feature vectors: " + e.getMessage());
            }
        }
    }

    private static void runSynthesis(FSPModel model, boolean verbose, Path file) {
        List<ControllerSpecDef> specs = model.controllerSpecs();
        if (specs.isEmpty()) {
            System.err.println("Must be at least a Goal to use synthesis");
            return;
        }
        ControllerSpecDef spec = specs.get(specs.size() - 1);

        Heuristic baseH;
        if (HEURISTIC == HeuristicType.RANDOM) {
            baseH = new RandomHeuristic(SCRIPT);
        } else if (HEURISTIC == HeuristicType.HAND || HEURISTIC == HeuristicType.SUPER_DFS) {
            String family = (file.getParent() != null) ? file.getParent().getFileName().toString() : "unknown";
            String fname = file.getFileName().toString();
            int n = 0, k = 0;
            if (fname.endsWith(".fsp")) {
                String[] parts = fname.substring(0, fname.length() - 4).split("-");
                if (parts.length >= 3) {
                    n = Integer.parseInt(parts[parts.length - 2]);
                    k = Integer.parseInt(parts[parts.length - 1]);
                }
            }
            baseH = HEURISTIC == HeuristicType.HAND
                    ? new HandHeuristic(family, n, k)
                    : new newmtsa.synthesis.heuristics.SuperDFSHeuristic(family, n, k);
        } else {
            baseH = HEURISTIC.create();
        }
        TraceRecordingHeuristic recorder = new TraceRecordingHeuristic(baseH);

        Director result;
        String mode;
        long startTime = System.nanoTime();
        if (spec.nonblocking()) {
            mode = "NonBlocking";
            System.out.println("\n--- Non-Blocking DCS: " + spec.name() + " ---");
            result = runNonBlocking(model, spec, verbose, VERBOSE, recorder);
        } else {
            mode = "GR1";
            System.out.println("\n--- GR(1) Synthesis: " + spec.name() + " ---");
            result = runGR1(model, spec, verbose, VERBOSE, recorder);
        }
        long elapsedMs = (System.nanoTime() - startTime) / 1_000_000;

        boolean realizable = result.isRealizable();
        System.out.println("  Result: " + (realizable ? "REALIZABLE" : "UNREALIZABLE"));

        System.out.println("  Expansions: " + recorder.getTrace().size());
        int directorTransitionCount = result.enabled().values().stream().mapToInt(List::size).sum();
        System.out.println("  Director transitions: " + directorTransitionCount);
        System.out.println("  Time: " + elapsedMs + " ms");

        try {
            if(SAVE_SOL) SolutionSaver.saveSol(file, result, mode, recorder.getTrace(), HEURISTIC, FEATURE_TYPE);
        } catch (IOException e) {
            System.err.println("WARNING: could not save .sol file: " + e.getMessage());
        }
    }

    private static Director runNonBlocking(FSPModel model, ControllerSpecDef spec, boolean verbose, boolean heuristicVerbose, Heuristic h) {
        List<LtlPropertyDef> safetyProps = new ArrayList<>();
        for (String name : spec.safety()) {
            model.ltlProperties().stream()
                    .filter(p -> p.name().equals(name))
                    .findFirst()
                    .ifPresent(safetyProps::add);
        }

        return new OTFDirectedControledSyntesisNonBlocking(
                new ArrayList<>(model.processes()),
                safetyProps,
                new HashSet<>(spec.marking()),
                new HashSet<>(spec.controllable()),
                h,
                verbose || heuristicVerbose,
                Integer.MAX_VALUE,
                USE_NUMERIC_STATE_IDS,
                null,
                FRONTIER_RESTRICTION
        ).run();
    }

    private static Director runGR1(FSPModel model, ControllerSpecDef spec, boolean verbose, boolean heuristicVerbose, Heuristic h) {
        List<LtlPropertyDef> guarantees = new ArrayList<>();
        for (String name : spec.liveness()) {
            model.asserts().stream()
                    .filter(p -> p.name().equals(name))
                    .findFirst()
                    .or(() -> model.fluents().stream()
                            .filter(f -> f.name().equals(name))
                            .map(f -> new LtlPropertyDef(f.name(), List.of(), f))
                            .findFirst())
                    .ifPresent(guarantees::add);
        }

        List<LtlPropertyDef> assumptions = new ArrayList<>();
        for (String name : spec.assumption()) {
            model.asserts().stream()
                    .filter(p -> p.name().equals(name))
                    .findFirst()
                    .or(() -> model.fluents().stream()
                            .filter(f -> f.name().equals(name))
                            .map(f -> new LtlPropertyDef(f.name(), List.of(), f))
                            .findFirst())
                    .ifPresent(assumptions::add);
        }

        List<LTS> components = new ArrayList<>(model.processes());
        for (LtlPropertyDef g : guarantees) {
            components.add(g.lts());
        }
        for (LtlPropertyDef a : assumptions) {
            if (components.stream().noneMatch(c -> c.name().equals(a.name()))) {
                components.add(a.lts());
            }
        }

        return new OTFDirectedControledSyntesisGR1(
                components, assumptions, guarantees,
                new HashSet<>(spec.controllable()),
                h,
                verbose || heuristicVerbose,
                USE_NUMERIC_STATE_IDS,
                null,
                FRONTIER_RESTRICTION
        ).run();
    }

    // ── feature vector collection ─────────────────────────────────────────────

    private static void saveFeatureVectors(FSPModel model, Path file) throws IOException {
        List<ControllerSpecDef> specs = model.controllerSpecs();
        if (specs.isEmpty()) { System.err.println("No controller spec — skipping feature vector save."); return; }
        ControllerSpecDef spec = specs.get(specs.size() - 1);

        int runs = (HEURISTIC == HeuristicType.RANDOM) ? RANDOM_RUNS : 1;

        Map<String, int[]>  vectorByKey = new LinkedHashMap<>();
        Map<String, Integer> countByKey = new LinkedHashMap<>();

        for (int run = 0; run < runs; run++) {
            Heuristic h = buildHeuristic(file);
            FeatureCompute fc = FeatureType.valueOf(FEATURE_TYPE.toUpperCase()).create();

            OTFDirectedControlledSynthesis dcs = spec.nonblocking()
                ? buildNBDCSWithFeatures(model, spec, h, fc)
                : buildGR1DCSWithFeatures(model, spec, h, fc);

            while (!dcs.isExplorationEnded()) {
                List<int[]> features = dcs.getFrontierWithFeatures();
                for (int[] fv : features) {
                    String key = Arrays.toString(fv);
                    countByKey.merge(key, 1, Integer::sum);
                    vectorByKey.putIfAbsent(key, fv);
                }
                List<ExtendedTransition> frontier = dcs.getFrontier();
                if (frontier.isEmpty()) break;
                int idx = h.pick(frontier);
                if (idx < 0 || idx >= frontier.size()) idx = 0;
                dcs.expand(idx);
            }
        }

        List<Map.Entry<String, Integer>> sorted = new ArrayList<>(countByKey.entrySet());
        sorted.sort((a, b) -> b.getValue() - a.getValue());

        String instanceName = file.getFileName().toString().replaceAll("\\.[^.]+$", "");
        String family       = (file.getParent() != null) ? file.getParent().getFileName().toString() : instanceName;
        Path   root         = findProjectRoot(file);
        Path   outDir       = root.resolve("python").resolve("results")
                                  .resolve("traces").resolve(family).resolve(HEURISTIC.name());
        java.nio.file.Files.createDirectories(outDir);
        Path outPath = outDir.resolve(instanceName + "_"
                           + FEATURE_TYPE.toLowerCase() + "_feature_vectors.txt");

        try (PrintWriter pw = new PrintWriter(new FileWriter(outPath.toFile()))) {
            pw.printf("instance=%s  heuristic=%s  feature=%s  runs=%d  unique_vectors=%d%n",
                file.getFileName(), HEURISTIC, FEATURE_TYPE, runs, sorted.size());
            pw.println("---");
            for (var entry : sorted) {
                int[] fv = vectorByKey.get(entry.getKey());
                java.math.BigInteger dec = java.math.BigInteger.ZERO;
                for (int bit : fv) dec = dec.shiftLeft(1).add(java.math.BigInteger.valueOf(bit));
                pw.printf("count=%-6d  decimal=%-20s  %s%n", entry.getValue(), dec, entry.getKey());
            }
        }
        System.out.println("Feature vectors saved to: " + outPath);
    }

    private static Heuristic buildHeuristic(Path file) {
        if (HEURISTIC == HeuristicType.RANDOM) return new RandomHeuristic(SCRIPT);
        if (HEURISTIC == HeuristicType.HAND) {
            String family = (file.getParent() != null) ? file.getParent().getFileName().toString() : "unknown";
            String fname  = file.getFileName().toString();
            int n = 0, k = 0;
            if (fname.endsWith(".fsp")) {
                String[] parts = fname.substring(0, fname.length() - 4).split("-");
                if (parts.length >= 3) { n = Integer.parseInt(parts[parts.length - 2]); k = Integer.parseInt(parts[parts.length - 1]); }
            }
            return new HandHeuristic(family, n, k);
        }
        return HEURISTIC.create();
    }

    private static OTFDirectedControlledSynthesis buildNBDCSWithFeatures(
            FSPModel model, ControllerSpecDef spec, Heuristic h, FeatureCompute fc) {
        List<LtlPropertyDef> safetyProps = new ArrayList<>();
        for (String name : spec.safety())
            model.ltlProperties().stream().filter(p -> p.name().equals(name)).findFirst().ifPresent(safetyProps::add);
        return new OTFDirectedControledSyntesisNonBlocking(
                new ArrayList<>(model.processes()), safetyProps,
                new HashSet<>(spec.marking()), new HashSet<>(spec.controllable()),
                h, false, Integer.MAX_VALUE, USE_NUMERIC_STATE_IDS, fc, FRONTIER_RESTRICTION);
    }

    private static Path findProjectRoot(Path knownPath) {
        Path p = knownPath.toAbsolutePath().normalize();
        if (!java.nio.file.Files.isDirectory(p)) p = p.getParent();
        while (p != null) {
            if (java.nio.file.Files.exists(p.resolve("pom.xml"))) return p;
            p = p.getParent();
        }
        return java.nio.file.Paths.get("").toAbsolutePath();
    }

    private static OTFDirectedControlledSynthesis buildGR1DCSWithFeatures(
            FSPModel model, ControllerSpecDef spec, Heuristic h, FeatureCompute fc) {
        List<LtlPropertyDef> guarantees = new ArrayList<>();
        for (String name : spec.liveness())
            model.asserts().stream().filter(p -> p.name().equals(name)).findFirst()
                .or(() -> model.fluents().stream().filter(f -> f.name().equals(name))
                    .map(f -> new LtlPropertyDef(f.name(), List.of(), f)).findFirst())
                .ifPresent(guarantees::add);
        List<LtlPropertyDef> assumptions = new ArrayList<>();
        for (String name : spec.assumption())
            model.asserts().stream().filter(p -> p.name().equals(name)).findFirst()
                .or(() -> model.fluents().stream().filter(f -> f.name().equals(name))
                    .map(f -> new LtlPropertyDef(f.name(), List.of(), f)).findFirst())
                .ifPresent(assumptions::add);
        List<LTS> components = new ArrayList<>(model.processes());
        for (LtlPropertyDef g : guarantees) components.add(g.lts());
        for (LtlPropertyDef a : assumptions)
            if (components.stream().noneMatch(c -> c.name().equals(a.name()))) components.add(a.lts());
        return new OTFDirectedControledSyntesisGR1(
                components, assumptions, guarantees, new HashSet<>(spec.controllable()),
                h, false, USE_NUMERIC_STATE_IDS, fc, FRONTIER_RESTRICTION);
    }
}
