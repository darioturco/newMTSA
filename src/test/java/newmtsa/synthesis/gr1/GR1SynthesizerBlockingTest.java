package newmtsa.synthesis.gr1;

import newmtsa.parser.FSPParser;
import newmtsa.parser.ast.ControllerSpecDef;
import newmtsa.parser.ast.FSPModel;
import newmtsa.parser.ast.LTS;
import newmtsa.parser.ast.LtlPropertyDef;
import newmtsa.synthesis.SynthesisResult;
import newmtsa.synthesis.heuristics.RandomHeuristic;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Parametrized GR(1) synthesis tests for every file under fsp/Blocking/.
 *
 * <p>Expected outcomes are inferred from the sub-directory name:
 * <ul>
 *   <li>{@code ControllableFSPs/}   → synthesis must return REALIZABLE.</li>
 *   <li>{@code NoControllableFSPs/} → synthesis must return UNREALIZABLE.</li>
 * </ul>
 */
class GR1SynthesizerBlockingTest {

    // ── file providers ────────────────────────────────────────────────────────

    static Stream<Path> controllableFiles() throws IOException {
        return Files.walk(Paths.get("fsp/Blocking/ControllableFSPs"))
                .filter(p -> p.toString().endsWith(".lts") || p.toString().endsWith(".fsp"));
    }

    static Stream<Path> noControllableFiles() throws IOException {
        return Files.walk(Paths.get("fsp/Blocking/NoControllableFSPs"))
                .filter(p -> p.toString().endsWith(".lts") || p.toString().endsWith(".fsp"));
    }

    // ── synthesis helper ──────────────────────────────────────────────────────

    /**
     * Locate the first controllerSpec with a non-empty liveness, build the
     * component list (processes + guarantee fluents), and run GR(1) synthesis
     * with a fixed random seed for reproducibility.
     */
    private SynthesisResult synthesize(FSPModel model) {
        ControllerSpecDef spec = model.controllerSpecs().stream()
                .filter(s -> !s.liveness().isEmpty())
                .findFirst()
                .orElseThrow(() -> new AssertionError("No liveness spec found in model"));

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

    // ── tests ─────────────────────────────────────────────────────────────────

    /**
     * Every file in ControllableFSPs must parse without errors and synthesize
     * to a REALIZABLE result (a valid director exists).
     */
    @ParameterizedTest(name = "{0}")
    @MethodSource("controllableFiles")
    void controllable_isRealizable(Path file) throws IOException {
        FSPModel model = FSPParser.parse(file);
        assertTrue(model.errors().isEmpty(),
                "Parse errors in " + file + ": " + model.errors());

        SynthesisResult result = synthesize(model);
        assertTrue(result.isRealizable(),
                file.getFileName() + " is in ControllableFSPs → must be REALIZABLE"
                        + " (states=" + result.statesExplored()
                        + ", transitions=" + result.transitionsExplored() + ")");
        assertTrue(result.director().isPresent(),
                "REALIZABLE result must carry a Director");
    }

    /**
     * Every file in NoControllableFSPs must parse without errors and synthesize
     * to UNREALIZABLE (no winning controller exists).
     */
    @ParameterizedTest(name = "{0}")
    @MethodSource("noControllableFiles")
    void noControllable_isUnrealizable(Path file) throws IOException {
        FSPModel model = FSPParser.parse(file);
        assertTrue(model.errors().isEmpty(),
                "Parse errors in " + file + ": " + model.errors());

        SynthesisResult result = synthesize(model);
        assertFalse(result.isRealizable(),
                file.getFileName() + " is in NoControllableFSPs → must be UNREALIZABLE"
                        + " (states=" + result.statesExplored()
                        + ", transitions=" + result.transitionsExplored() + ")");
    }
}
