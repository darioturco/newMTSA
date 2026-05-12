package newmtsa.synthesis.gr1;

import newmtsa.parser.FSPParser;
import newmtsa.parser.ast.*;
import newmtsa.synthesis.heuristics.HeuristicType;
import newmtsa.synthesis.heuristics.ra.RAHeuristic;
import newmtsa.synthesis.heuristics.ra.EstimateTuple;
import newmtsa.synthesis.heuristics.SimpleSynthesisContext;
import newmtsa.synthesis.ExtendedTransition;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.*;

class RA_Estimate_Test {
    @Test
    void traceFirstPick() throws Exception {
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
            HeuristicType.RA_ERG_OPEN.create(),
            false, true);

        RAHeuristic h = (RAHeuristic) e.heuristic;
        for (int step = 1; step <= 4; step++) {
            List<ExtendedTransition> f = e.getFrontier();
            int idx = h.pick(f);
            System.out.println("=== Step " + step + " pick=[" + idx + "] " + f.get(idx).action() + " from " + f.get(idx).from() + " ===");
            for (int i = 0; i < f.size(); i++) {
                ExtendedTransition t = f.get(i);
                List<EstimateTuple> est = h.estimateCacheFor(t.from(), t.action());
                System.out.println("  [" + i + "] " + t.from() + " --" + t.action() + "--> " + t.to() + " est=" + est);
            }
            e.expand(idx);
        }
    }
}
