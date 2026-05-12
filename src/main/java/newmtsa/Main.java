package newmtsa;

import newmtsa.parser.FSPParser;
import newmtsa.parser.ast.*;
import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.Director;
import newmtsa.synthesis.gr1.OTFDirectedControledSyntesisGR1;
import newmtsa.synthesis.heuristics.Heuristic;
import newmtsa.synthesis.heuristics.HeuristicType;
import newmtsa.synthesis.heuristics.RandomHeuristic;
import newmtsa.synthesis.heuristics.SynthesisContext;
import newmtsa.synthesis.nonblocking.OTFDirectedControledSyntesisNonBlocking;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;


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
    //static final HeuristicType HEURISTIC = HeuristicType.RA_OPEN;
    static final HeuristicType HEURISTIC = HeuristicType.RA_ERG_OPEN;
    //static final HeuristicType HEURISTIC = HeuristicType.RL;
    //static final HeuristicType HEURISTIC = HeuristicType.MCTS_RL;

    // Print composite states as names (false) or numeric indices (true).
    static final boolean USE_NUMERIC_STATE_IDS = true;

    static final boolean VERBOSE = false; // Print heuristic frontier and transition scores before each expansion (RA shows distances).
    static final boolean SAVE_SOL = true; // Flag to save the trace in a .sol file

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

        List<TraceStep> getTrace() { return trace; }
    }

    // ── main ──────────────────────────────────────────────────────────────────

    public static void main(String[] args) throws IOException {
        Path file;
        if(args.length > 0){
            file = Path.of(args[0]);
        }else{
            //file = Path.of(".\\fsp\\NonBlocking\\Benchmark\\CM\\CM-2-2.fsp");
            //file = Path.of(".\\fsp\\NonBlocking\\Benchmark\\DP\\DP-2-2.fsp");
            file = Path.of(".\\fsp\\Blocking\\Benchmark\\DP\\DP-1-1.fsp");
            //file = Path.of(".\\fsp\\NonBlocking\\Benchmark\\TA\\TA-2-2.fsp");
            //file = Path.of(".\\fsp\\NonBlocking\\Benchmark\\AT\\AT-2-2.fsp");
            //file = Path.of(".\\fsp\\NonBlocking\\Benchmark\\BW\\BW-2-2.fsp");
            //file = Path.of(".\\fsp\\NonBlocking\\ControllableFSPs\\test21.lts");
            //file = Path.of(".\\fsp\\Blocking\\ControllableFSPs\\GR1Test43.lts");
            //file = Path.of(".\\fsp\\NonBlocking\\Benchmark\\CM\\CM-2-2.fsp");
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

        runSynthesis(model, true, file);
    }

    private static void runSynthesis(FSPModel model, boolean verbose, Path file) {
        List<ControllerSpecDef> specs = model.controllerSpecs();
        if (specs.isEmpty()) {
            System.err.println("Must be at least a Goal to use synthesis");
            return;
        }
        ControllerSpecDef spec = specs.get(specs.size() - 1);

        Heuristic baseH = (HEURISTIC == HeuristicType.RANDOM) ? new RandomHeuristic(SCRIPT) : HEURISTIC.create();
        TraceRecordingHeuristic recorder = new TraceRecordingHeuristic(baseH);

        Director result;
        String mode;
        if (spec.nonblocking()) {
            mode = "NonBlocking";
            System.out.println("\n--- Non-Blocking DCS: " + spec.name() + " ---");
            result = runNonBlocking(model, spec, verbose, VERBOSE, recorder);
        } else {
            mode = "GR1";
            System.out.println("\n--- GR(1) Synthesis: " + spec.name() + " ---");
            result = runGR1(model, spec, verbose, VERBOSE, recorder);
        }

        System.out.println("  Result: " + (result.isRealizable() ? "REALIZABLE" : "UNREALIZABLE"));

        try {
            if(SAVE_SOL) saveSol(file, result, mode, recorder.getTrace());
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
                USE_NUMERIC_STATE_IDS
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
                USE_NUMERIC_STATE_IDS
        ).run();
    }

    // ── .sol file writer ──────────────────────────────────────────────────────

    private static void saveSol(Path file, Director result, String mode, List<TraceStep> trace) throws IOException {
        String fileName = file.getFileName().toString();
        String instanceName = fileName.contains(".")
                ? fileName.substring(0, fileName.lastIndexOf('.'))
                : fileName;
        String family = (file.getParent() != null) ? file.getParent().getFileName().toString() : "unknown";

        Path outDir;
        if (HEURISTIC == HeuristicType.RL || HEURISTIC == HeuristicType.MCTS_RL) {
            outDir = Path.of("python", "results", "traces", family, HEURISTIC.name(), FEATURE_TYPE);
        } else {
            outDir = Path.of("python", "results", "traces", family, HEURISTIC.name());
        }
        Files.createDirectories(outDir);
        Path outFile = outDir.resolve(instanceName + ".sol");

        try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(outFile))) {
            pw.println(instanceName);
            pw.println("Trace Len: " + trace.size());
            pw.println("Realizable: " + (result.isRealizable() ? "True" : "False"));
            pw.println("Mode: " + mode);
            pw.println("Heuristic Used: " + HEURISTIC.name());
            pw.println("Heuristic Parameters: " + heuristicParams());
            pw.println("Trace: ");
            pw.println();
            for (TraceStep step : trace) {
                pw.println(formatFrontier(step.frontier()));
                pw.println(step.index() + " | " + formatTransition(step.selected()));
            }
            if (result.isRealizable()) {
                pw.println();
                pw.println("Director Transitions:");
                pw.println();
                for (var entry : result.enabled().entrySet()) {
                    for (String action : entry.getValue()) {
                        pw.println("  " + entry.getKey() + " --[" + action + "]-->");
                    }
                }
            }
        }

        System.out.println("  Saved: " + outFile);
    }

    private static String heuristicParams() {
        return switch (HEURISTIC) {
            case RANDOM  -> "SCRIPT=" + SCRIPT;
            case RL      -> "model_path=" + MODEL_PATH + ", feature_type=" + FEATURE_TYPE;
            case MCTS_RL -> "model_path=" + MODEL_PATH + ", feature_type=" + FEATURE_TYPE
                    + ", simulations=" + System.getProperty("mcts.simulations", "50")
                    + ", cpuct=" + System.getProperty("mcts.cpuct", "1.5")
                    + ", depth=" + System.getProperty("mcts.depth", "10");
            default -> "(none)";
        };
    }

    private static String formatTransition(ExtendedTransition t) {
        return t.from() + " --[" + t.action() + "]--> " + t.to();
    }

    private static String formatFrontier(List<ExtendedTransition> frontier) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < frontier.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(formatTransition(frontier.get(i)));
        }
        sb.append("]");
        return sb.toString();
    }
}
