package newmtsa.synthesis.gr1;

import newmtsa.parser.FSPParser;
import newmtsa.parser.ast.*;
import newmtsa.synthesis.SynthesisResult;
import newmtsa.synthesis.heuristics.HeuristicType;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

/** Debug test that runs TL-2-2 with verbose open queue output. */
class TL22DebugTest {

    @Test
    void debugAllVariants() throws Exception {
        Path file = Path.of("./fsp/Blocking/Benchmark/TL/TL-2-2.fsp");
        FSPModel model = FSPParser.parse(file);

        ControllerSpecDef spec = model.controllerSpecs().get(model.controllerSpecs().size() - 1);
        if (spec.nonblocking()) return;

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

        List<LTS> components = new ArrayList<>(model.processes());
        for (LtlPropertyDef g : guarantees) components.add(g.lts());

        for (HeuristicType type : new HeuristicType[]{
                HeuristicType.RA, HeuristicType.RA_OPEN,
                HeuristicType.RA_E, HeuristicType.RA_E_OPEN,
                HeuristicType.RA_ER, HeuristicType.RA_ER_OPEN,
                HeuristicType.RA_ERG, HeuristicType.RA_ERG_OPEN}) {
            OTFDirectedControledSyntesisGR1 engine = new OTFDirectedControledSyntesisGR1(
                    components, assumptions, guarantees,
                    new HashSet<>(spec.controllable()),
                    type.create(),
                    false, true);
            SynthesisResult result = engine.run();
            System.out.println(type + ": transitions=" + engine.getTransitionsExplored()
                + " states=" + engine.getStatesExplored()
                + " realizable=" + result.isRealizable());
        }
    }
}
