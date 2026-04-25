package newmtsa.synthesis.nonblocking;

import newmtsa.parser.FSPParser;
import newmtsa.parser.ast.ControllerSpecDef;
import newmtsa.parser.ast.FSPModel;
import newmtsa.parser.ast.LtlPropertyDef;
import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.heuristics.Heuristic;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Replays 10 expansion steps for each benchmark family (2-2 instance) using the
 * same action sequence as the reference MTSA implementation and asserts that
 * the frontier size before each pick matches the expected values.
 *
 * <p>Data source: {@code ToDo.md} (collected from the old MTSA JAR).
 *
 * <p>Action notation: the reference uses dot-notation ({@code get.0});
 * our synthesis uses bracket-notation ({@code get[0]}).
 * {@link ActionScriptedHeuristic} converts automatically.
 */
class FamilyFrontierSizeTest {

    // ── test cases ────────────────────────────────────────────────────────────

    record FamilyCase(
            String name,
            String fspPath,
            List<String> script,   // dot-notation
            int[]        expected  // frontier sizes before each pick
    ) {}

    static Stream<FamilyCase> cases() {
        return Stream.of(
            new FamilyCase("TL-2-2",
                "fsp/NonBlocking/Benchmark/TL/TL-2-2.fsp",
                List.of("get.0","put.1","get.1","put.2","get.2",
                        "return.1","accept","reject","get.1","get.0"),
                new int[]{3,4,6,9,11,14,16,18,20,19}),

            new FamilyCase("AT-2-2",
                "fsp/NonBlocking/Benchmark/AT/AT-2-2.fsp",
                List.of("requestLand.0","requestLand.1","extendFlight.0","requestLand.1",
                        "extendFlight.1","requestLand.0","extendFlight.1","control.all",
                        "requestLand.0","requestLand.1"),
                new int[]{4,5,6,7,8,9,10,10,13,12}),

            new FamilyCase("BW-2-2",
                "fsp/NonBlocking/Benchmark/BW/BW-2-2.fsp",
                List.of("assign.0","reject.0.1","accept.0","assign.0","accept.0",
                        "reject.0.2","assign.1","reject.1.1","accept.1","approve"),
                new int[]{4,6,9,12,14,13,16,15,18,20}),

            new FamilyCase("CM-2-2",
                "fsp/NonBlocking/Benchmark/CM/CM-2-2.fsp",
                List.of("mouse.turn","mouse.1.move.3","mouse.0.move.3","cat.turn",
                        "cat.0.move.0","cat.0.move.1","cat.1.move.0","cat.1.move.1",
                        "cat.1.move.0","cat.0.move.0"),
                new int[]{1,4,5,5,8,9,10,11,12,12}),

            new FamilyCase("DP-2-2",
                "fsp/NonBlocking/Benchmark/DP/DP-2-2.fsp",
                List.of("think.0","think.1","think.1","think.0","take.0.0",
                        "step.0","step.0","take.0.1","eat.0","release.0.0"),
                new int[]{2,3,4,5,4,5,6,7,7,7}),

            new FamilyCase("TA-2-2",
                "fsp/NonBlocking/Benchmark/TA/TA-2-2.fsp",
                List.of("agency.request","query.0","available.0","steps.0.0",
                        "query.succ.0","steps.0.1","query.succ.0","select.0",
                        "select.0","unavailable.0"),
                new int[]{1,3,2,3,3,3,3,3,2,3})
        );
    }

    // ── test ──────────────────────────────────────────────────────────────────

    @ParameterizedTest(name = "{0}")
    @MethodSource("cases")
    void frontierSizesMatchReference(FamilyCase fc) throws IOException {
        FSPModel model = FSPParser.parse(Path.of(fc.fspPath()));
        assertTrue(model.errors().isEmpty(),
                "Parse errors in " + fc.fspPath() + ": " + model.errors());

        ControllerSpecDef spec = model.controllerSpecs().stream()
                .filter(ControllerSpecDef::nonblocking)
                .reduce((a, b) -> b)
                .orElseThrow(() -> new AssertionError("No nonblocking controllerSpec in " + fc.name()));

        List<LtlPropertyDef> safetyProps = new ArrayList<>();
        for (String name : spec.safety())
            model.ltlProperties().stream()
                    .filter(p -> p.name().equals(name))
                    .findFirst()
                    .ifPresent(safetyProps::add);

        ActionScriptedHeuristic h = new ActionScriptedHeuristic(fc.script());
        int steps = fc.expected().length;

        new OTFDirectedControledSyntesisNonBlocking(
                new ArrayList<>(model.processes()),
                safetyProps,
                new HashSet<>(spec.marking()),
                new HashSet<>(spec.controllable()),
                h,
                false,
                steps
        ).run();

        assertEquals(steps, h.frontierSizes.size(),
                fc.name() + ": expected " + steps + " frontier snapshots");

        for (int i = 0; i < steps; i++) {
            int actual   = h.frontierSizes.get(i);
            int expected = fc.expected()[i];
            assertEquals(expected, actual,
                    fc.name() + " step " + (i + 1)
                    + ": expected frontier size " + expected + " but got " + actual);
        }
    }

    // ── ActionScriptedHeuristic ───────────────────────────────────────────────

    /**
     * Heuristic that records open-queue frontier sizes and picks scripted actions.
     *
     * <p>Script entries are in dot-notation ({@code get.0}); they are converted
     * to bracket-notation ({@code get[0]}) for matching against the frontier.
     *
     * <p>Frontier sizes reflect the <em>open queue</em>: transitions from states
     * whose remaining pending transitions are all controllable.  This matches the
     * semantics of the original MTSA implementation (Ciolek §5.1.3).
     */
    static class ActionScriptedHeuristic implements Heuristic {

        final List<Integer> frontierSizes = new ArrayList<>();
        private final List<String> scriptBracket;
        private int step = 0;
        private Set<String> controllable = Set.of();

        // Stateful open-queue tracking (mirrors RAHeuristic.openQueueCandidates).
        private final Set<String> closedStates = new HashSet<>();
        private ExtendedTransition lastPicked = null;

        ActionScriptedHeuristic(List<String> scriptDot) {
            this.scriptBracket = new ArrayList<>();
            for (String entry : scriptDot) scriptBracket.add(dotToBracket(entry));
        }

        @Override
        public void init(newmtsa.synthesis.heuristics.SynthesisContext ctx) {
            this.controllable = ctx.controllable();
        }

        /**
         * Computes the open-queue size using the same stateful logic as
         * {@code RAHeuristic.openQueueCandidates}: close lastPicked.from() if it
         * still has uncontrollable transitions, reopen closed states that no longer
         * have uncontrollable transitions in pending, then count open transitions.
         */
        private int openQueueSize(List<ExtendedTransition> pending) {
            // Close the source of the last picked transition if it has uncontrollable remaining.
            if (lastPicked != null) {
                String src = lastPicked.from();
                for (ExtendedTransition t : pending) {
                    if (t.from().equals(src) && !controllable.contains(t.action())) {
                        closedStates.add(src);
                        break;
                    }
                }
            }
            // Reopen closed states whose uncontrollable transitions have all left pending.
            closedStates.removeIf(s -> {
                for (ExtendedTransition t : pending)
                    if (t.from().equals(s) && !controllable.contains(t.action())) return false;
                return true;
            });
            // Count transitions from open states.
            int count = 0;
            for (ExtendedTransition t : pending)
                if (!closedStates.contains(t.from())) count++;
            return count;
        }

        @Override
        public ExtendedTransition pick(List<ExtendedTransition> pending) {
            frontierSizes.add(openQueueSize(pending));

            ExtendedTransition chosen = null;
            if (step < scriptBracket.size()) {
                String want = scriptBracket.get(step++);
                for (ExtendedTransition t : pending) {
                    if (t.action().equals(want)) { chosen = t; break; }
                }
                if (chosen == null) {
                    System.err.printf(
                            "[ActionScriptedHeuristic] step %d: scripted action '%s' not in frontier"
                            + " (size=%d) — falling back to first.%n",
                            step, want, pending.size());
                }
            } else {
                step++;
            }
            if (chosen == null) chosen = pending.get(0);
            lastPicked = chosen;
            return chosen;
        }

        /**
         * Converts dot-notation to bracket-notation:
         * {@code "mouse.1.move.3"} → {@code "mouse[1].move[3]"}
         */
        static String dotToBracket(String dot) {
            String[] parts = dot.split("\\.");
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < parts.length; i++) {
                if (i == 0) {
                    sb.append(parts[i]);
                } else if (parts[i].matches("\\d+")) {
                    sb.append('[').append(parts[i]).append(']');
                } else {
                    sb.append('.').append(parts[i]);
                }
            }
            return sb.toString();
        }
    }
}
