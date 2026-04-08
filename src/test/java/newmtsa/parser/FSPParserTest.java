package newmtsa.parser;

import newmtsa.parser.ast.ControllerSpecDef;
import newmtsa.parser.ast.FSPModel;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

class FSPParserTest {

    static Stream<Path> allFSPFiles() throws IOException {
        Path fspDir = Paths.get("fsp");
        return Files.walk(fspDir)
                .filter(p -> p.toString().endsWith(".lts") || p.toString().endsWith(".fsp"));
    }

    // --- Parse-success tests (one per file) ---

    @ParameterizedTest(name = "{0}")
    @MethodSource("allFSPFiles")
    void parsesWithoutErrors(Path file) throws IOException {
        FSPModel model = FSPParser.parse(file);
        assertNotNull(model);
        assertTrue(
                model.errors().isEmpty(),
                "Parse errors in " + file + ": " + model.errors()
        );
    }

    // --- Specific structure tests ---

    /**
     * fsp/Blocking/ControllableFSPs/GR1test1.lts
     *
     * Expected top-level definitions:
     *  - 1 process:        Ejemplo
     *  - 3 composites:     Plant, DirectedController (heuristic), Sys
     *  - 2 fluents:        F1, F2
     *  - 3 asserts:        S1, S2, Check
     *  - 1 controllerSpec: Goal  (liveness = {S1, S2}, controllable = {a})
     */
    @Test
    void gr1test1_hasCorrectTopLevelDefinitions() throws IOException {
        Path file = Paths.get("fsp/Blocking/ControllableFSPs/GR1test1.lts");
        FSPModel model = FSPParser.parse(file);

        assertEquals(1, model.processes().size());
        assertEquals("Ejemplo", model.processes().get(0).name());

        assertEquals(3, model.composites().size());
        assertTrue(model.composites().stream().anyMatch(c -> c.name().equals("Plant")));
        assertTrue(model.composites().stream().anyMatch(c -> c.name().equals("DirectedController")));
        assertTrue(model.composites().stream().anyMatch(c -> c.name().equals("Sys")));

        assertEquals(2, model.fluents().size());
        assertTrue(model.fluents().stream().anyMatch(f -> f.name().equals("F1")));
        assertTrue(model.fluents().stream().anyMatch(f -> f.name().equals("F2")));

        assertEquals(3, model.asserts().size());
        assertTrue(model.asserts().stream().anyMatch(a -> a.name().equals("S1")));
        assertTrue(model.asserts().stream().anyMatch(a -> a.name().equals("S2")));
        assertTrue(model.asserts().stream().anyMatch(a -> a.name().equals("Check")));

        assertEquals(1, model.controllerSpecs().size());
        ControllerSpecDef goal = model.controllerSpecs().get(0);
        assertEquals("Goal", goal.name());
        assertTrue(goal.liveness().contains("S1"));
        assertTrue(goal.liveness().contains("S2"));
        assertTrue(goal.controllable().contains("a"));
    }

    /**
     * fsp/NonBlocking/ControllableFSPs/test1.lts
     *
     * controllerSpec Goal has:
     *  - controllable = {c45}
     *  - marking      = {u55}
     *  - nonblocking  flag set
     */
    @Test
    void nonBlockingTest1_controllerSpecHasMarkingAndNonblocking() throws IOException {
        Path file = Paths.get("fsp/NonBlocking/ControllableFSPs/test1.lts");
        FSPModel model = FSPParser.parse(file);

        assertEquals(1, model.controllerSpecs().size());
        ControllerSpecDef spec = model.controllerSpecs().get(0);
        assertEquals("Goal", spec.name());
        assertTrue(spec.controllable().contains("c45"));
        assertTrue(spec.marking().contains("u55"));
        assertTrue(spec.nonblocking());
    }

    /**
     * fsp/Benchmark/DP/DP-2-2.fsp  — Dining Philosophers benchmark
     *
     * Expected top-level definitions:
     *  - 3 processes:      Philosopher, Fork, Monitor
     *  - 3 composites:     Plant, MonolithicController, DirectedController
     *  - 1 controllerSpec: Goal  (controllable = {take[...]}, marking = {eat.all}, nonblocking)
     */
    @Test
    void benchmarkDP_parsesWithoutErrors() throws IOException {
        Path file = Paths.get("fsp/Benchmark/DP/DP-2-2.fsp");
        FSPModel model = FSPParser.parse(file);
        assertTrue(
                model.errors().isEmpty(),
                "Parse errors in DP-2-2.fsp: " + model.errors()
        );
    }

    @Test
    void benchmarkDP_hasCorrectTopLevelDefinitions() throws IOException {
        Path file = Paths.get("fsp/Benchmark/DP/DP-2-2.fsp");
        FSPModel model = FSPParser.parse(file);

        // Phil = 0..1 → 2 instances of each of the 3 process templates = 6 instances.
        assertEquals(6, model.processes().size());
        assertTrue(model.processes().stream().anyMatch(p -> p.name().equals("Philosopher(0)")));
        assertTrue(model.processes().stream().anyMatch(p -> p.name().equals("Philosopher(1)")));
        assertTrue(model.processes().stream().anyMatch(p -> p.name().equals("Fork(0)")));
        assertTrue(model.processes().stream().anyMatch(p -> p.name().equals("Fork(1)")));
        assertTrue(model.processes().stream().anyMatch(p -> p.name().equals("Monitor(0)")));
        assertTrue(model.processes().stream().anyMatch(p -> p.name().equals("Monitor(1)")));

        assertEquals(3, model.composites().size());
        assertTrue(model.composites().stream().anyMatch(c -> c.name().equals("Plant")));
        assertTrue(model.composites().stream().anyMatch(c -> c.name().equals("MonolithicController")));
        assertTrue(model.composites().stream().anyMatch(c -> c.name().equals("DirectedController")));

        assertEquals(1, model.controllerSpecs().size());
        ControllerSpecDef goal = model.controllerSpecs().get(0);
        assertEquals("Goal", goal.name());
        assertTrue(goal.nonblocking());
    }

    /**
     * fsp/Benchmark/TL/TL-2-2.fsp  — Transfer Line benchmark
     *
     * Expected top-level definitions:
     *  - 3 processes:      Machine, TU, Buffer
     *  - 3 composites:     Plant, MonolithicController, DirectedController
     *  - 1 controllerSpec: Goal  (marking = {accept, reject}, nonblocking)
     */
    @Test
    void benchmarkTL_parsesWithoutErrors() throws IOException {
        Path file = Paths.get("fsp/Benchmark/TL/TL-2-2.fsp");
        FSPModel model = FSPParser.parse(file);
        assertTrue(
                model.errors().isEmpty(),
                "Parse errors in TL-2-2.fsp: " + model.errors()
        );
    }

    @Test
    void benchmarkTL_hasCorrectTopLevelDefinitions() throws IOException {
        Path file = Paths.get("fsp/Benchmark/TL/TL-2-2.fsp");
        FSPModel model = FSPParser.parse(file);

        assertEquals(5, model.processes().size());
        assertTrue(model.processes().stream().anyMatch(p -> p.name().equals("Machine(0)")));
        assertTrue(model.processes().stream().anyMatch(p -> p.name().equals("Machine(1)")));
        assertTrue(model.processes().stream().anyMatch(p -> p.name().equals("Buffer(1)")));
        assertTrue(model.processes().stream().anyMatch(p -> p.name().equals("Buffer(2)")));
        assertTrue(model.processes().stream().anyMatch(p -> p.name().equals("TU")));

        assertEquals(3, model.composites().size());
        assertTrue(model.composites().stream().anyMatch(c -> c.name().equals("Plant")));
        assertTrue(model.composites().stream().anyMatch(c -> c.name().equals("MonolithicController")));
        assertTrue(model.composites().stream().anyMatch(c -> c.name().equals("DirectedController")));

        assertEquals(1, model.controllerSpecs().size());
        ControllerSpecDef goal = model.controllerSpecs().get(0);
        assertEquals("Goal", goal.name());
        assertTrue(goal.marking().contains("accept"));
        assertTrue(goal.marking().contains("reject"));
        assertTrue(goal.nonblocking());
    }

    /**
     * fsp/Benchmark/CM/CM-2-2.fsp  — Cats and Mice benchmark
     *
     * Verifies operator-precedence correctness:
     *   const Areas = 2*Levels+1  with Levels=1 (K=1)  →  (2*1)+1 = 3,  NOT 2*(1+1)=4
     *   const Center = Areas \ 2                        →  3\2 = 1
     *   const Last   = Areas-1                          →  2
     */
    @Test
    void benchmarkCM_constantsHaveCorrectValues() throws IOException {
        Path file = Paths.get("fsp/Benchmark/CM/CM-2-2.fsp");
        FSPModel model = FSPParser.parse(file);

        assertTrue(model.errors().isEmpty(),
                "Parse errors in CM-2-2.fsp: " + model.errors());

        var consts = model.constants();
        java.util.function.Function<String, Integer> get =
                name -> consts.stream()
                        .filter(c -> c.name().equals(name))
                        .mapToInt(c -> c.value()).findFirst().orElseThrow();

        assertEquals(1, get.apply("Levels"), "Levels");
        assertEquals(3, get.apply("Areas"),  "Areas = 2*Levels+1 must respect * before +");
        assertEquals(1, get.apply("Center"), "Center = Areas \\ 2");
        assertEquals(2, get.apply("Last"),   "Last = Areas-1");
    }

    /**
     * fsp/NonBlocking/NoControllableFSPs/test14.lts
     *
     * Has:
     *  - 1 ltl_property: Nunca_Loop_No_Controlable
     *  - controllerSpec Goal with safety = {Nunca_Loop_No_Controlable} and nonblocking
     */
    @Test
    void nonBlockingNoControllableTest14_hasLtlPropertyAndSafety() throws IOException {
        Path file = Paths.get("fsp/NonBlocking/NoControllableFSPs/test14.lts");
        FSPModel model = FSPParser.parse(file);

        assertEquals(1, model.ltlProperties().size());
        assertEquals("Nunca_Loop_No_Controlable", model.ltlProperties().get(0).name());

        assertEquals(1, model.controllerSpecs().size());
        ControllerSpecDef spec = model.controllerSpecs().get(0);
        assertTrue(spec.safety().contains("Nunca_Loop_No_Controlable"));
        assertTrue(spec.nonblocking());
    }
}
