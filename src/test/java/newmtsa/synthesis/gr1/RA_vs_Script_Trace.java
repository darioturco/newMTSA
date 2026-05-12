package newmtsa.synthesis.gr1;

import newmtsa.parser.FSPParser;
import newmtsa.parser.ast.*;
import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.heuristics.Heuristic;
import newmtsa.synthesis.heuristics.HeuristicType;
import newmtsa.synthesis.heuristics.RandomHeuristic;
import newmtsa.synthesis.heuristics.ra.EstimateTuple;
import newmtsa.synthesis.heuristics.ra.RAHeuristic;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.*;

/** Side-by-side trace of RA_ERG_OPEN vs scripted RANDOM (= old's 16-step plan). */
class RA_vs_Script_Trace {

    static final List<Integer> SCRIPT =
        List.of(0, 3, 6, 9, 12, 14, 17, 18, 14, 15, 22, 25, 27, 22, 28, 17);

    record StepRec(int step, int idx, String from, String action, String to,
                   List<ExtendedTransition> frontier) {}

    private static List<StepRec> runHeuristic(Heuristic h, int maxSteps) throws Exception {
        Path file = Path.of("./fsp/Blocking/Benchmark/TL/TL-2-2.fsp");
        FSPModel model = FSPParser.parse(file);
        ControllerSpecDef spec = model.controllerSpecs().get(model.controllerSpecs().size() - 1);
        List<LtlPropertyDef> guarantees = new ArrayList<>();
        for (String name : spec.liveness()) {
            model.asserts().stream().filter(p -> p.name().equals(name)).findFirst()
                .or(() -> model.fluents().stream().filter(f -> f.name().equals(name))
                    .map(f -> new LtlPropertyDef(f.name(), List.of(), f)).findFirst())
                .ifPresent(guarantees::add);
        }
        List<LtlPropertyDef> assumptions = new ArrayList<>();
        List<LTS> components = new ArrayList<>(model.processes());
        for (LtlPropertyDef g : guarantees) components.add(g.lts());
        OTFDirectedControledSyntesisGR1 e = new OTFDirectedControledSyntesisGR1(
            components, assumptions, guarantees,
            new HashSet<>(spec.controllable()),
            h, false, true);

        List<StepRec> recs = new ArrayList<>();
        int step = 1;
        while (!e.isExplorationEnded() && step <= maxSteps) {
            List<ExtendedTransition> f = e.getFrontier();
            if (f.isEmpty()) break;
            int idx = h.pick(f);
            ExtendedTransition t = f.get(idx);
            recs.add(new StepRec(step, idx, t.from(), t.action(), t.to(), new ArrayList<>(f)));
            e.expand(idx);
            if (e.getGoals().contains(t.from()) && f.size() == 1) break;
            step++;
        }
        return recs;
    }

    @Test
    void dumpStep1Estimates() throws Exception {
        Path file = Path.of("./fsp/Blocking/Benchmark/TL/TL-2-2.fsp");
        FSPModel model = FSPParser.parse(file);
        ControllerSpecDef spec = model.controllerSpecs().get(model.controllerSpecs().size() - 1);
        List<LtlPropertyDef> guarantees = new ArrayList<>();
        for (String name : spec.liveness()) {
            model.asserts().stream().filter(p -> p.name().equals(name)).findFirst()
                .or(() -> model.fluents().stream().filter(f -> f.name().equals(name))
                    .map(f -> new LtlPropertyDef(f.name(), List.of(), f)).findFirst())
                .ifPresent(guarantees::add);
        }
        List<LtlPropertyDef> assumptions = new ArrayList<>();
        List<LTS> components = new ArrayList<>(model.processes());
        for (LtlPropertyDef g : guarantees) components.add(g.lts());
        RAHeuristic h = (RAHeuristic) HeuristicType.RA_ERG_OPEN.create();
        OTFDirectedControledSyntesisGR1 e = new OTFDirectedControledSyntesisGR1(
            components, assumptions, guarantees,
            new HashSet<>(spec.controllable()),
            h, false, true);

        List<ExtendedTransition> f = e.getFrontier();
        int picked = h.pick(f);
        System.out.println("\n=== Initial frontier RA estimates ===");
        for (int i = 0; i < f.size(); i++) {
            ExtendedTransition t = f.get(i);
            List<EstimateTuple> est = h.estimateCacheFor(t.from(), t.action());
            String tag = (i == picked) ? " <-- pick" : "";
            System.out.printf("  [%2d] %-8s %s%n", i, t.action(), est + tag);
        }
        System.out.println("Component count = " + components.size());
        for (int j = 0; j < components.size(); j++) {
            LTS lts = components.get(j);
            System.out.println("  comp " + j + ": " + lts.name()
                + " initial=" + lts.initialState()
                + " marked=" + lts.acceptingStates()
                + " #states=" + lts.states().size());
            if (lts.name().equals("F") || j == 5) {
                System.out.println("    F transitions:");
                for (Transition t : lts.transitions()) {
                    System.out.println("      " + t.from() + " --[" + t.action() + "]--> " + t.to());
                }
            }
        }
    }

    @Test
    void compare() throws Exception {
        List<StepRec> ra = runHeuristic(HeuristicType.RA_ERG_OPEN.create(), 30);
        List<StepRec> sc = runHeuristic(new RandomHeuristic(SCRIPT), 30);

        int n = Math.max(ra.size(), sc.size());
        System.out.println("\n========================== STEP-BY-STEP COMPARISON ==========================");
        System.out.printf("%-4s | %-50s | %-50s%n", "Step", "RA_ERG_OPEN", "SCRIPT (old's 16 picks)");
        System.out.println("-".repeat(110));
        boolean diverged = false;
        for (int i = 0; i < n; i++) {
            String left  = i < ra.size() ? fmt(ra.get(i)) : "(done)";
            String right = i < sc.size() ? fmt(sc.get(i)) : "(done)";
            String mark = " ";
            if (i < ra.size() && i < sc.size()) {
                StepRec a = ra.get(i), b = sc.get(i);
                boolean same = a.action.equals(b.action) && a.from.equals(b.from);
                if (!same && !diverged) { mark = "*"; diverged = true; }
                else if (!same) mark = "x";
                else mark = " ";
            }
            System.out.printf("%-3s%s | %-50s | %-50s%n", (i + 1), mark, left, right);
        }
        System.out.println("\nRA total: " + ra.size() + "   SCRIPT total: " + sc.size());

        // Detail: dump full frontier at first divergence
        for (int i = 0; i < Math.min(ra.size(), sc.size()); i++) {
            StepRec a = ra.get(i), b = sc.get(i);
            if (!a.action.equals(b.action) || !a.from.equals(b.from)) {
                System.out.println("\n=== FIRST DIVERGENCE at step " + (i + 1) + " ===");
                System.out.println("RA picked idx=" + a.idx + " : " + a.from + " --[" + a.action + "]--> " + a.to);
                System.out.println("SCRIPT picked idx=" + b.idx + " : " + b.from + " --[" + b.action + "]--> " + b.to);
                System.out.println("\nRA frontier:");
                for (int k = 0; k < a.frontier.size(); k++) {
                    ExtendedTransition t = a.frontier.get(k);
                    String tag = (k == a.idx) ? " <- RA pick" : "";
                    System.out.printf("  [%2d] %s --[%s]--> %s%s%n", k, t.from(), t.action(), t.to(), tag);
                }
                System.out.println("\nSCRIPT frontier:");
                for (int k = 0; k < b.frontier.size(); k++) {
                    ExtendedTransition t = b.frontier.get(k);
                    String tag = (k == b.idx) ? " <- SCRIPT pick" : "";
                    System.out.printf("  [%2d] %s --[%s]--> %s%s%n", k, t.from(), t.action(), t.to(), tag);
                }
                break;
            }
        }
    }

    private static String fmt(StepRec r) {
        return String.format("[%2d] %s --[%s]-> %s",
            r.idx, abbr(r.from, 18), abbr(r.action, 8), abbr(r.to, 18));
    }

    private static String abbr(String s, int max) {
        return s.length() <= max ? s : s.substring(0, max - 2) + "..";
    }
}
