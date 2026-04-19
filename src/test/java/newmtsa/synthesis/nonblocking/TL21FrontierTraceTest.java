package newmtsa.synthesis.nonblocking;

import newmtsa.parser.FSPParser;
import newmtsa.parser.ast.*;
import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.heuristics.Heuristic;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Replays the first 8 expansion steps of TL-2-1 from the reference trace
 * (tmp_exp.txt) and verifies that the frontier at each step matches exactly
 * the expected set of (from-state, action) pairs.
 *
 * <p><b>Composite state format</b> (sub-states joined by "|"):
 * <pre>
 *   Machine(0) | Buffer(1) | Machine(1) | Buffer(2) | TU | marking_fluent
 * </pre>
 *
 * <p><b>Action naming</b>: bracket notation — {@code get[0]}, {@code put[1]}, etc.
 *
 * <p><b>Why certain actions must be absent</b>:
 * TU carries {@code +{return[0..Machines]}} in the FSP, which forces
 * {@code return[0]}, {@code return[1]}, {@code return[2]} into TU's alphabet
 * even though TU has no transition on them from {@code Idle}.  This means those
 * actions are BLOCKED whenever TU is at {@code Idle}, preventing them from
 * appearing in the frontier via Buffer(1)/Buffer(2) alone.  If this alphabet
 * extension is missing from the parser, spurious {@code return} transitions will
 * appear — the test will flag them as unexpected entries in the frontier.
 */
class TL21FrontierTraceTest {

    // ── composite state constants ─────────────────────────────────────────────
    //   format: Machine(0)|Buffer(1)|Machine(1)|Buffer(2)|TU|fluent

    /** Initial state – all components at their initial sub-state. */
    static final String S0 =
            "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off";

    /** After get[0] from S0: Machine(0) → Working[1]. */
    static final String S1 =
            "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off";

    /** After put[1] from S1: Machine(0) → Working[0], Buffer(1) → Operative[1]. */
    static final String S5 =
            "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off";

    /** After get[1] from S5: Buffer(1) → Operative[0], Machine(1) → Working[1]. */
    static final String S6 =
            "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off";

    /** After get[0] from S5: Machine(0) → Working[1]. */
    static final String S7 =
            "Working[1]|Operative[1]|Working[0]|Operative[0]|Idle|off";

    // ── helpers ───────────────────────────────────────────────────────────────

    /** Encodes a (state, action) pair as a single string for set membership. */
    private static String ta(String state, String action) {
        return state + "||" + action;
    }

    /** Builds an expected frontier set from vararg (state||action) encoded strings. */
    private static Set<String> fr(String... entries) {
        return new LinkedHashSet<>(Arrays.asList(entries));
    }

    // ── expected frontiers ────────────────────────────────────────────────────

    /**
     * Frontier content (order-independent set) before each of the 8 scripted picks.
     *
     * <p>Derived by manually tracing the DCS algorithm with the same sequence of
     * transitions as the reference implementation (tmp_exp.txt).
     */
    static final List<Set<String>> EXPECTED = List.of(

        // Step 1 – initial frontier: 3 controllable transitions from S0.
        fr( ta(S0,"get[0]"), ta(S0,"get[1]"), ta(S0,"get[2]") ),

        // Step 2 – after S0 –get[0]→ S1 [NONE]:
        //   remaining from S0 + 3 transitions newly added from S1.
        fr( ta(S0,"get[1]"), ta(S0,"get[2]"),
            ta(S1,"get[1]"), ta(S1,"get[2]"), ta(S1,"put[1]") ),

        // Step 3 – after S0 –get[2]→ ERROR:
        //   (S0,"get[2]") consumed; no new transitions added (error sink).
        fr( ta(S0,"get[1]"),
            ta(S1,"get[1]"), ta(S1,"get[2]"), ta(S1,"put[1]") ),

        // Step 4 – after S0 –get[1]→ ERROR:
        fr( ta(S1,"get[1]"), ta(S1,"get[2]"), ta(S1,"put[1]") ),

        // Step 5 – after S1 –get[2]→ ERROR:
        fr( ta(S1,"get[1]"), ta(S1,"put[1]") ),

        // Step 6 – after S1 –put[1]→ S5 [NONE]:
        //   3 transitions from S5 added.
        fr( ta(S1,"get[1]"),
            ta(S5,"get[0]"), ta(S5,"get[1]"), ta(S5,"get[2]") ),

        // Step 7 – after S5 –get[1]→ S6 [NONE]:
        //   3 transitions from S6 added; (S5,"get[1]") consumed.
        fr( ta(S1,"get[1]"),
            ta(S5,"get[0]"), ta(S5,"get[2]"),
            ta(S6,"get[0]"), ta(S6,"get[2]"), ta(S6,"put[2]") ),

        // Step 8 – after S5 –get[0]→ S7 [NONE]:
        //   3 transitions from S7 added; (S5,"get[0]") consumed.
        fr( ta(S1,"get[1]"),
            ta(S5,"get[2]"),
            ta(S6,"get[0]"), ta(S6,"get[2]"), ta(S6,"put[2]"),
            ta(S7,"get[1]"), ta(S7,"get[2]"), ta(S7,"put[1]") )
    );

    // ── expansion script ──────────────────────────────────────────────────────

    /**
     * The (from-state, action) pair to pick at each step, mirroring the
     * reference trace in tmp_exp.txt.
     */
    static final List<String[]> SCRIPT = List.of(
        new String[]{ S0, "get[0]" },  // step 1 → S1  [NONE]
        new String[]{ S0, "get[2]" },  // step 2 → ERROR
        new String[]{ S0, "get[1]" },  // step 3 → ERROR
        new String[]{ S1, "get[2]" },  // step 4 → ERROR
        new String[]{ S1, "put[1]" },  // step 5 → S5  [NONE]
        new String[]{ S5, "get[1]" },  // step 6 → S6  [NONE]
        new String[]{ S5, "get[0]" },  // step 7 → S7  [NONE]
        new String[]{ S7, "put[1]" }   // step 8 → ERROR (budget exhausted after this)
    );

    // ── capturing heuristic ───────────────────────────────────────────────────

    /**
     * Heuristic that:
     * <ol>
     *   <li>Records the full frontier (as a set of {@code "state||action"} strings)
     *       before every pick.</li>
     *   <li>Picks the transition matching the scripted (state, action) at that step.
     *       If not found, falls back to the first pending transition and prints a
     *       warning — the frontier assertion at that step will then likely fail,
     *       helping diagnose which state names differ.</li>
     *   <li>After the script is exhausted, always picks the first pending transition.</li>
     * </ol>
     */
    static class CapturingHeuristic implements Heuristic {
        final List<Set<String>> frontiers = new ArrayList<>();
        private int step = 0;

        @Override
        public ExtendedTransition pick(List<ExtendedTransition> pending) {
            frontiers.add(pending.stream()
                    .map(t -> t.from() + "||" + t.action())
                    .collect(Collectors.toCollection(LinkedHashSet::new)));

            if (step < SCRIPT.size()) {
                String wantState  = SCRIPT.get(step)[0];
                String wantAction = SCRIPT.get(step)[1];
                step++;
                for (ExtendedTransition t : pending)
                    if (t.from().equals(wantState) && t.action().equals(wantAction))
                        return t;
                System.err.printf(
                        "[CapturingHeuristic] step %d: scripted (%s, %s) not in frontier"
                        + " — falling back to first pending transition.%n",
                        step, wantState, wantAction);
            } else {
                step++;
            }
            return pending.get(0);
        }
    }

    // ── test ──────────────────────────────────────────────────────────────────

    @Test
    void tl21_frontierMatchesReferenceFor8Steps() throws IOException {
        FSPModel model = FSPParser.parse(Path.of("fsp/NonBlocking/Benchmark/TL/TL-2-1.fsp"));
        assertTrue(model.errors().isEmpty(), "Parse errors: " + model.errors());

        ControllerSpecDef spec = model.controllerSpecs().stream()
                .filter(ControllerSpecDef::nonblocking)
                .reduce((a, b) -> b)
                .orElseThrow(() -> new AssertionError("No nonblocking controllerSpec found"));

        List<LtlPropertyDef> safetyProps = new ArrayList<>();
        for (String name : spec.safety())
            model.ltlProperties().stream()
                    .filter(p -> p.name().equals(name))
                    .findFirst()
                    .ifPresent(safetyProps::add);

        CapturingHeuristic heuristic = new CapturingHeuristic();

        new OTFDirectedControledSyntesisNonBlocking(
                new ArrayList<>(model.processes()),
                safetyProps,
                new HashSet<>(spec.marking()),
                new HashSet<>(spec.controllable()),
                heuristic,
                false,
                8   // stop after exactly 8 expansions
        ).run();

        assertEquals(8, heuristic.frontiers.size(),
                "Expected exactly 8 frontier snapshots (one per expansion step)");

        for (int i = 0; i < EXPECTED.size(); i++) {
            Set<String> expected = EXPECTED.get(i);
            Set<String> actual   = heuristic.frontiers.get(i);
            assertEquals(expected, actual,
                    "Frontier mismatch at step " + (i + 1)
                    + "\n  extra   (in actual but not expected): " + minus(actual, expected)
                    + "\n  missing (in expected but not actual): " + minus(expected, actual));
        }
    }

    // ── helper ────────────────────────────────────────────────────────────────

    private static Set<String> minus(Set<String> a, Set<String> b) {
        Set<String> r = new LinkedHashSet<>(a);
        r.removeAll(b);
        return r;
    }
}
