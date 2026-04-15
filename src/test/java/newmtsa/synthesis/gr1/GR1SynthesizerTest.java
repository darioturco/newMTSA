package newmtsa.synthesis.gr1;

import newmtsa.parser.FSPParser;
import newmtsa.parser.ast.ControllerSpecDef;
import newmtsa.parser.ast.FSPModel;
import newmtsa.parser.ast.LTS;
import newmtsa.parser.ast.LtlPropertyDef;
import newmtsa.synthesis.Director;
import newmtsa.synthesis.SynthesisResult;
import newmtsa.synthesis.heuristics.RandomHeuristic;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class GR1SynthesizerTest {

    private SynthesisResult synthesize(FSPModel model) {
        ControllerSpecDef spec = model.controllerSpecs().stream()
                .filter(s -> !s.liveness().isEmpty())
                .findFirst()
                .orElseThrow(() -> new AssertionError("No liveness spec found"));

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
                new RandomHeuristic(42L),
                false
        ).run();
    }

    /**
     * GR1Test1: simple 1-process, 2-fluent GR(1) problem.
     * Expected: REALIZABLE.
     */
    @Test
    void gr1test1_isRealizable() throws IOException {
        FSPModel model = FSPParser.parse(Paths.get("fsp/Blocking/ControllableFSPs/GR1test1.lts"));
        assertTrue(model.errors().isEmpty(), "Parse errors: " + model.errors());

        SynthesisResult result = synthesize(model);
        assertTrue(result.isRealizable(), "GR1test1 should be realizable");
        assertNotNull(result.director().orElse(null));
    }

    /**
     * GR1Test27: 4 guarantee fluents (FA, FB, FC, FF after assert resolution).
     * Expected: REALIZABLE.
     */
    @Test
    void gr1test27_isRealizable() throws IOException {
        FSPModel model = FSPParser.parse(Paths.get("fsp/Blocking/ControllableFSPs/GR1Test27.lts"));
        assertTrue(model.errors().isEmpty(), "Parse errors: " + model.errors());

        SynthesisResult result = synthesize(model);
        assertTrue(result.isRealizable(), "GR1Test27 should be realizable");
    }

    /**
     * Director has enabled actions for the initial extended state.
     */
    @Test
    void gr1test1_directorCoversInitialState() throws IOException {
        FSPModel model = FSPParser.parse(Paths.get("fsp/Blocking/ControllableFSPs/GR1test1.lts"));
        SynthesisResult result = synthesize(model);
        assertTrue(result.isRealizable());
        Director d = result.director().get();
        // Director must be non-empty
        assertFalse(d.enabled().isEmpty(), "Director should have at least one state entry");
    }
}
