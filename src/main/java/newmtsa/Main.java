package newmtsa;

import newmtsa.parser.FSPParser;
import newmtsa.parser.ast.*;
import newmtsa.synthesis.SynthesisResult;
import newmtsa.synthesis.gr1.OTFDirectedControledSyntesisGR1;
import newmtsa.synthesis.heuristics.Heuristic;
import newmtsa.synthesis.heuristics.HeuristicType;
import newmtsa.synthesis.heuristics.RandomHeuristic;
import newmtsa.synthesis.nonblocking.OTFDirectedControledSyntesisNonBlocking;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import java.io.IOException;
import java.nio.file.Path;


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
    //static final HeuristicType HEURISTIC = HeuristicType.RA_ERG;
    //static final HeuristicType HEURISTIC = HeuristicType.RANDOM;
    //static final HeuristicType HEURISTIC = HeuristicType.FIRST;
    //static final HeuristicType HEURISTIC = HeuristicType.BFS;
    //static final HeuristicType HEURISTIC = HeuristicType.HUMAN;
    //static final HeuristicType HEURISTIC = HeuristicType.RA_R;
    //static final HeuristicType HEURISTIC = HeuristicType.RA_E;
    //static final HeuristicType HEURISTIC = HeuristicType.RA_ER;
    static final HeuristicType HEURISTIC = HeuristicType.RA_ERG;
    //static final HeuristicType HEURISTIC = HeuristicType.RA_OPEN;
    //static final HeuristicType HEURISTIC = HeuristicType.RA_ERG_OPEN;

    // Print composite states as names (false) or numeric indices (true).
    static final boolean USE_NUMERIC_STATE_IDS = true;

    // Scripted frontier indices used when HEURISTIC = RANDOM.
    // At step k picks pending.get(SCRIPT[k]); falls back to random once exhausted.
    // List.of() = pure random from the start.
    // [0,0,0,1] selects actions a[1], a[3], a[2], a[8] for FloppyTesis.fsp.
    //static final List<Integer> SCRIPT = List.of(0, 3, 6, 9, 12, 14, 17, 18, 14, 15, 22, 25, 27, 22, 28, 17);   // unused when HEURISTIC != RANDOM (Expansion to solve TL-2-2 in the same way the original RA should.)
    static final List<Integer> SCRIPT = List.of();

    public static void main(String[] args) throws IOException {
        Path file;
        if(args.length > 0){
            file = Path.of(args[0]);
        }else{
            //file = Path.of(".\\fsp\\NonBlocking\\Benchmark\\CM\\CM-2-2.fsp");
            //file = Path.of(".\\fsp\\NonBlocking\\Benchmark\\DP\\DP-2-2.fsp");
            file = Path.of(".\\fsp\\NonBlocking\\Benchmark\\TL\\TL-2-2.fsp");
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

        runSynthesis(model, true);
    }

    private static void runSynthesis(FSPModel model, boolean verbose) {
        List<ControllerSpecDef> specs = model.controllerSpecs();
        if (specs.isEmpty()) {
            System.err.println("Must be at least a Goal to use synthesis");
            return;
        }
        ControllerSpecDef spec = specs.get(specs.size() - 1);

        SynthesisResult result;
        if (spec.nonblocking()) {
            System.out.println("\n--- Non-Blocking DCS: " + spec.name() + " ---");
            result = runNonBlocking(model, spec, verbose);
        } else {
            System.out.println("\n--- GR(1) Synthesis: " + spec.name() + " ---");
            result = runGR1(model, spec, verbose);
        }

        System.out.println("  Result: " + (result.isRealizable() ? "REALIZABLE" : "UNREALIZABLE"));
        System.out.println("  Termination reason  : " + result.terminationReason());
        System.out.println("  Transitions expanded: " + result.transitionsExplored());
    }

    private static SynthesisResult runNonBlocking(FSPModel model, ControllerSpecDef spec, boolean verbose) {
        List<LtlPropertyDef> safetyProps = new ArrayList<>();
        for (String name : spec.safety()) {
            model.ltlProperties().stream()
                    .filter(p -> p.name().equals(name))
                    .findFirst()
                    .ifPresent(safetyProps::add);
        }

        Heuristic h;
        if (HEURISTIC == HeuristicType.RANDOM) {
            h = new RandomHeuristic(SCRIPT);
        } else {
            h = HEURISTIC.create();
        }

        return new OTFDirectedControledSyntesisNonBlocking(
                new ArrayList<>(model.processes()),
                safetyProps,
                new HashSet<>(spec.marking()),
                new HashSet<>(spec.controllable()),
                h,
                verbose,
                Integer.MAX_VALUE,
                USE_NUMERIC_STATE_IDS
        ).run();
    }

    private static SynthesisResult runGR1(FSPModel model, ControllerSpecDef spec, boolean verbose) {
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
                HEURISTIC.create(),
                verbose,
                USE_NUMERIC_STATE_IDS
        ).run();
    }
}
