package newmtsa.synthesis.gr1;

import newmtsa.parser.FSPParser;
import newmtsa.parser.ast.*;
import newmtsa.synthesis.heuristics.HeuristicType;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

class RA_ERG_OPEN_Trace {
    @Test
    void trace() throws Exception {
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
            true, true);
        e.run();
        System.out.println("Total transitions: " + e.getTransitionsExplored());
    }
}
