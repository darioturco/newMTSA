package newmtsa.synthesis.gr1;

import newmtsa.parser.FSPParser;
import newmtsa.parser.ast.*;
import newmtsa.synthesis.heuristics.RandomHeuristic;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.*;

class RA_Script_Trace {
    @Test
    void scriptTrace() throws Exception {
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
        List<Integer> SCRIPT = List.of(0, 3, 6, 9, 12, 14, 17, 18, 14, 15, 22, 25, 27, 22, 28, 17);
        OTFDirectedControledSyntesisGR1 e = new OTFDirectedControledSyntesisGR1(
            components, assumptions, guarantees,
            new HashSet<>(spec.controllable()),
            new RandomHeuristic(SCRIPT),
            true, true);
        e.run();
        System.out.println("Total transitions: " + e.getTransitionsExplored());
    }
}
