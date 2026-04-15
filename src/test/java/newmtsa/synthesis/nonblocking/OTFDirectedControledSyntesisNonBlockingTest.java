package newmtsa.synthesis.nonblocking;

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
 * Parametrized non-blocking DCS tests for every file under fsp/NonBlocking/.
 *
 * <p>Expected outcomes are inferred from the sub-directory name:
 * <ul>
 *   <li>{@code ControllableFSPs/}   → synthesis must return REALIZABLE.</li>
 *   <li>{@code NoControllableFSPs/} → synthesis must return UNREALIZABLE.</li>
 * </ul>
 */
class OTFDirectedControledSyntesisNonBlockingTest {

    // ── file providers ────────────────────────────────────────────────────────

    static Stream<Path> controllableFiles() throws IOException {
        return Files.walk(Paths.get("fsp/NonBlocking/ControllableFSPs"))
                .filter(p -> p.toString().endsWith(".lts") || p.toString().endsWith(".fsp"));
    }

    static Stream<Path> noControllableFiles() throws IOException {
        return Files.walk(Paths.get("fsp/NonBlocking/NoControllableFSPs"))
                .filter(p -> p.toString().endsWith(".lts") || p.toString().endsWith(".fsp"));
    }

    // ── synthesis helper ──────────────────────────────────────────────────────

    /**
     * Locate the last controllerSpec with {@code nonblocking = true}, build the
     * component list from the model's processes, and run non-blocking DCS with a
     * fixed random seed for reproducibility.
     */
    private SynthesisResult synthesize(FSPModel model) {
        List<ControllerSpecDef> specs = model.controllerSpecs();
        ControllerSpecDef spec = null;
        for (int i = specs.size() - 1; i >= 0; i--) {
            if (specs.get(i).nonblocking()) { spec = specs.get(i); break; }
        }
        if (spec == null) throw new AssertionError("No nonblocking spec found in model");

        // Collect safety properties named in the spec's safety list.
        List<LtlPropertyDef> safetyProps = new ArrayList<>();
        for (String name : spec.safety()) {
            model.ltlProperties().stream()
                    .filter(p -> p.name().equals(name))
                    .findFirst()
                    .ifPresent(safetyProps::add);
        }

        List<LTS> components = new ArrayList<>(model.processes());

        return new OTFDirectedControledSyntesisNonBlocking(
                components,
                safetyProps,
                new HashSet<>(spec.marking()),
                new HashSet<>(spec.controllable()),
                new RandomHeuristic(42L),
                false
        ).run();
    }

    // ── tests ─────────────────────────────────────────────────────────────────

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
