package newmtsa.synthesis.gr1;

import newmtsa.synthesis.DCSForPython;
import newmtsa.synthesis.ExtendedTransition;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

class DCSFrontierRegressionTest {

    record Case(String label, String path, int[] trace, List<List<String>> snapshots) {
        @Override public String toString() { return label; }
    }

    private static List<String> snap(List<ExtendedTransition> f) {
        List<String> s = new ArrayList<>(f.size());
        for (ExtendedTransition t : f) s.add(t.from() + "|" + t.action() + "|" + t.to());
        return s;
    }

    private static void assertTrace(Case c) throws Exception {
        DCSForPython dcs = DCSForPython.fromPath(c.path(), "FIRST", "");
        for (int step = 0; step < c.snapshots().size(); step++) {
            assertFalse(dcs.isExplorationEnded(), "ended early at step " + step);
            List<ExtendedTransition> frontier = dcs.getFrontier();
            assertEquals(c.snapshots().get(step), snap(frontier),
                "frontier mismatch step=" + step + " in " + c.label());
            int idx = c.trace() == null ? 0 : c.trace()[step];
            dcs.expand(idx);
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("controllableCases")
    void controllable_frontierRegression(Case c) throws Exception { assertTrace(c); }

    @ParameterizedTest(name = "{0}")
    @MethodSource("noControllableCases")
    void noControllable_frontierRegression(Case c) throws Exception { assertTrace(c); }

    // TODO: fix the test (dont do, ask for human help)
    // @Test void tl22_frontierRegression() throws Exception { assertTrace(caseTL22()); }

    static Stream<Case> controllableCases() { return Stream.of(
        caseGR1Test3(), caseGR1Test5(), caseGR1Test7(), caseGR1Test10(),
        caseGR1Test13(), caseGR1Test14(), caseGR1Test15(), caseGR1Test16(),
        caseGR1Test22(), caseGR1Test24()); }

    static Stream<Case> noControllableCases() { return Stream.of(
        caseGR1Test4(), caseGR1Test6(), caseGR1Test11(), caseGR1Test12(),
        caseGR1Test17(), caseGR1Test18(), caseGR1Test19(), caseGR1Test20(),
        caseGR1Test21(), caseGR1Test23()); }

    // ── CONTROLLABLE ──────────────────────────────────────────────────────────

    private static Case caseGR1Test3() {
        return new Case("GR1Test3", "fsp/Blocking/ControllableFSPs/GR1Test3.lts", null, List.of(
            List.of("A0|off|off#0|a|A1|on|off#0"),
            List.of("A1|on|off#0|b|A2|off|off#1", "A1|on|off#0|c|A3|off|off#1"),
            List.of("A1|on|off#0|c|A3|off|off#1", "A2|off|off#1|d|A0|off|on#1"),
            List.of("A2|off|off#1|d|A0|off|on#1", "A3|off|off#1|e|A0|off|off#1"),
            List.of("A3|off|off#1|e|A0|off|off#1", "A0|off|on#1|a|A1|on|off#0"),
            List.of("A0|off|on#1|a|A1|on|off#0", "A0|off|off#1|a|A1|on|off#1")
        ));
    }

    private static Case caseGR1Test5() {
        return new Case("GR1Test5", "fsp/Blocking/ControllableFSPs/GR1Test5.lts", null, List.of(
            List.of("A0|off_off#0|a|A1|off_off#0"),
            List.of("A1|off_off#0|b|A2|off_off#0", "A1|off_off#0|c|A3|off_off#0"),
            List.of("A1|off_off#0|c|A3|off_off#0", "A2|off_off#0|d|A0|off_on#0"),
            List.of("A2|off_off#0|d|A0|off_on#0", "A3|off_off#0|e|A0|on_off#0"),
            List.of("A3|off_off#0|e|A0|on_off#0", "A0|off_on#0|a|A1|off_off#0"),
            List.of("A0|off_on#0|a|A1|off_off#0", "A0|on_off#0|a|A1|off_off#0"),
            List.of("A0|on_off#0|a|A1|off_off#0")
        ));
    }

    private static Case caseGR1Test7() {
        return new Case("GR1Test7", "fsp/Blocking/ControllableFSPs/GR1Test7.lts", null, List.of(
            List.of("A0|off_off|off#0|a|A1|on_off|off#0", "A0|off_off|off#0|b|A1|off_off|off#0"),
            List.of("A0|off_off|off#0|b|A1|off_off|off#0", "A1|on_off|off#0|c|A2|on_on|off#0", "A1|on_off|off#0|d|A3|on_off|off#0"),
            List.of("A1|on_off|off#0|c|A2|on_on|off#0", "A1|on_off|off#0|d|A3|on_off|off#0", "A1|off_off|off#0|c|A2|off_on|off#0", "A1|off_off|off#0|d|A3|off_off|off#0"),
            List.of("A1|on_off|off#0|d|A3|on_off|off#0", "A1|off_off|off#0|c|A2|off_on|off#0", "A1|off_off|off#0|d|A3|off_off|off#0", "A2|on_on|off#0|e|A0|off_off|off#0"),
            List.of("A1|off_off|off#0|c|A2|off_on|off#0", "A1|off_off|off#0|d|A3|off_off|off#0", "A2|on_on|off#0|e|A0|off_off|off#0", "A3|on_off|off#0|f|A0|on_off|on#0"),
            List.of("A1|off_off|off#0|d|A3|off_off|off#0", "A2|on_on|off#0|e|A0|off_off|off#0", "A3|on_off|off#0|f|A0|on_off|on#0", "A2|off_on|off#0|e|A0|off_off|off#1"),
            List.of("A2|on_on|off#0|e|A0|off_off|off#0", "A3|on_off|off#0|f|A0|on_off|on#0", "A2|off_on|off#0|e|A0|off_off|off#1", "A3|off_off|off#0|f|A0|off_off|on#0"),
            List.of("A3|on_off|off#0|f|A0|on_off|on#0", "A2|off_on|off#0|e|A0|off_off|off#1", "A3|off_off|off#0|f|A0|off_off|on#0"),
            List.of("A2|off_on|off#0|e|A0|off_off|off#1", "A3|off_off|off#0|f|A0|off_off|on#0", "A0|on_off|on#0|a|A1|on_off|off#0", "A0|on_off|on#0|b|A1|on_off|off#0"),
            List.of("A3|off_off|off#0|f|A0|off_off|on#0", "A0|on_off|on#0|a|A1|on_off|off#0", "A0|on_off|on#0|b|A1|on_off|off#0", "A0|off_off|off#1|a|A1|on_off|off#1", "A0|off_off|off#1|b|A1|off_off|off#1"),
            List.of("A0|on_off|on#0|a|A1|on_off|off#0", "A0|on_off|on#0|b|A1|on_off|off#0", "A0|off_off|off#1|a|A1|on_off|off#1", "A0|off_off|off#1|b|A1|off_off|off#1", "A0|off_off|on#0|a|A1|on_off|off#0", "A0|off_off|on#0|b|A1|off_off|off#0"),
            List.of("A0|on_off|on#0|b|A1|on_off|off#0", "A0|off_off|off#1|a|A1|on_off|off#1", "A0|off_off|off#1|b|A1|off_off|off#1", "A0|off_off|on#0|a|A1|on_off|off#0", "A0|off_off|on#0|b|A1|off_off|off#0")
        ));
    }

    private static Case caseGR1Test10() {
        return new Case("GR1Test10", "fsp/Blocking/ControllableFSPs/GR1Test10.lts", null, List.of(
            List.of("A0|off|off#0|a|A1|off|off#0", "A0|off|off#0|c|A2|off|off#0"),
            List.of("A0|off|off#0|c|A2|off|off#0", "A1|off|off#0|b|A0|on|off#0"),
            List.of("A1|off|off#0|b|A0|on|off#0", "A2|off|off#0|d|A0|off|on#0"),
            List.of("A2|off|off#0|d|A0|off|on#0", "A0|on|off#0|a|A1|off|off#1", "A0|on|off#0|c|A2|off|off#1"),
            List.of("A0|on|off#0|a|A1|off|off#1", "A0|on|off#0|c|A2|off|off#1", "A0|off|on#0|a|A1|off|off#0", "A0|off|on#0|c|A2|off|off#0"),
            List.of("A0|on|off#0|c|A2|off|off#1", "A0|off|on#0|a|A1|off|off#0", "A0|off|on#0|c|A2|off|off#0", "A1|off|off#1|b|A0|on|off#1"),
            List.of("A0|off|on#0|a|A1|off|off#0", "A0|off|on#0|c|A2|off|off#0", "A1|off|off#1|b|A0|on|off#1", "A2|off|off#1|d|A0|off|on#1"),
            List.of("A0|off|on#0|c|A2|off|off#0", "A1|off|off#1|b|A0|on|off#1", "A2|off|off#1|d|A0|off|on#1"),
            List.of("A1|off|off#1|b|A0|on|off#1", "A2|off|off#1|d|A0|off|on#1"),
            List.of("A2|off|off#1|d|A0|off|on#1", "A0|on|off#1|a|A1|off|off#1", "A0|on|off#1|c|A2|off|off#1"),
            List.of("A0|on|off#1|a|A1|off|off#1", "A0|on|off#1|c|A2|off|off#1", "A0|off|on#1|a|A1|off|off#0", "A0|off|on#1|c|A2|off|off#0"),
            List.of("A0|on|off#1|c|A2|off|off#1", "A0|off|on#1|a|A1|off|off#0", "A0|off|on#1|c|A2|off|off#0")
        ));
    }

    private static Case caseGR1Test13() {
        return new Case("GR1Test13", "fsp/Blocking/ControllableFSPs/GR1Test13.lts", null, List.of(
            List.of("A1|off|off|off|off#0|a|A2|off|on|off|off#0"),
            List.of("A2|off|on|off|off#0|b|A3|off|off|on|off#0"),
            List.of("A3|off|off|on|off#0|c|A4|off|off|off|on#0", "A3|off|off|on|off#0|e|A1|on|off|off|off#0"),
            List.of("A3|off|off|on|off#0|e|A1|on|off|off|off#0", "A4|off|off|off|on#0|d|A2|off|off|off|off#0"),
            List.of("A4|off|off|off|on#0|d|A2|off|off|off|off#0", "A1|on|off|off|off#0|a|A2|off|on|off|off#1"),
            List.of("A1|on|off|off|off#0|a|A2|off|on|off|off#1", "A2|off|off|off|off#0|b|A3|off|off|on|off#0"),
            List.of("A2|off|off|off|off#0|b|A3|off|off|on|off#0", "A2|off|on|off|off#1|b|A3|off|off|on|off#2"),
            List.of("A2|off|on|off|off#1|b|A3|off|off|on|off#2"),
            List.of("A3|off|off|on|off#2|c|A4|off|off|off|on#3", "A3|off|off|on|off#2|e|A1|on|off|off|off#3"),
            List.of("A3|off|off|on|off#2|e|A1|on|off|off|off#3", "A4|off|off|off|on#3|d|A2|off|off|off|off#0"),
            List.of("A4|off|off|off|on#3|d|A2|off|off|off|off#0", "A1|on|off|off|off#3|a|A2|off|on|off|off#3")
        ));
    }

    private static Case caseGR1Test14() {
        return new Case("GR1Test14", "fsp/Blocking/ControllableFSPs/GR1Test14.lts", null, List.of(
            List.of("A0|off_off|off#0|a|A1|on_off|off#0", "A0|off_off|off#0|b|A1|off_off|off#0"),
            List.of("A0|off_off|off#0|b|A1|off_off|off#0", "A1|on_off|off#0|c|A2|on_on|off#0", "A1|on_off|off#0|d|A3|on_off|off#0"),
            List.of("A1|on_off|off#0|c|A2|on_on|off#0", "A1|on_off|off#0|d|A3|on_off|off#0", "A1|off_off|off#0|c|A2|off_on|off#0", "A1|off_off|off#0|d|A3|off_off|off#0"),
            List.of("A1|on_off|off#0|d|A3|on_off|off#0", "A1|off_off|off#0|c|A2|off_on|off#0", "A1|off_off|off#0|d|A3|off_off|off#0", "A2|on_on|off#0|e|A0|off_off|off#0"),
            List.of("A1|off_off|off#0|c|A2|off_on|off#0", "A1|off_off|off#0|d|A3|off_off|off#0", "A2|on_on|off#0|e|A0|off_off|off#0", "A3|on_off|off#0|f|A0|on_off|on#0"),
            List.of("A1|off_off|off#0|d|A3|off_off|off#0", "A2|on_on|off#0|e|A0|off_off|off#0", "A3|on_off|off#0|f|A0|on_off|on#0", "A2|off_on|off#0|e|A0|off_off|off#1"),
            List.of("A2|on_on|off#0|e|A0|off_off|off#0", "A3|on_off|off#0|f|A0|on_off|on#0", "A2|off_on|off#0|e|A0|off_off|off#1", "A3|off_off|off#0|f|A0|off_off|on#0"),
            List.of("A3|on_off|off#0|f|A0|on_off|on#0", "A2|off_on|off#0|e|A0|off_off|off#1", "A3|off_off|off#0|f|A0|off_off|on#0"),
            List.of("A2|off_on|off#0|e|A0|off_off|off#1", "A3|off_off|off#0|f|A0|off_off|on#0", "A0|on_off|on#0|a|A1|on_off|off#0", "A0|on_off|on#0|b|A1|on_off|off#0"),
            List.of("A3|off_off|off#0|f|A0|off_off|on#0", "A0|on_off|on#0|a|A1|on_off|off#0", "A0|on_off|on#0|b|A1|on_off|off#0", "A0|off_off|off#1|a|A1|on_off|off#1", "A0|off_off|off#1|b|A1|off_off|off#1"),
            List.of("A0|on_off|on#0|a|A1|on_off|off#0", "A0|on_off|on#0|b|A1|on_off|off#0", "A0|off_off|off#1|a|A1|on_off|off#1", "A0|off_off|off#1|b|A1|off_off|off#1", "A0|off_off|on#0|a|A1|on_off|off#0", "A0|off_off|on#0|b|A1|off_off|off#0"),
            List.of("A0|on_off|on#0|b|A1|on_off|off#0", "A0|off_off|off#1|a|A1|on_off|off#1", "A0|off_off|off#1|b|A1|off_off|off#1", "A0|off_off|on#0|a|A1|on_off|off#0", "A0|off_off|on#0|b|A1|off_off|off#0")
        ));
    }

    private static Case caseGR1Test15() {
        return new Case("GR1Test15", "fsp/Blocking/ControllableFSPs/GR1Test15.lts", null, List.of(
            List.of("A0|off|off|off#0|a|A1|off|off|off#0"),
            List.of("A1|off|off|off#0|b|A2|off|off|off#0", "A1|off|off|off#0|d|A3|on|off|off#0"),
            List.of("A1|off|off|off#0|d|A3|on|off|off#0", "A2|off|off|off#0|c|A1|off|off|off#0"),
            List.of("A2|off|off|off#0|c|A1|off|off|off#0", "A3|on|off|off#0|e|A4|off|on|off#1", "A3|on|off|off#0|f|A5|off|off|on#1"),
            List.of("A3|on|off|off#0|e|A4|off|on|off#1", "A3|on|off|off#0|f|A5|off|off|on#1"),
            List.of("A3|on|off|off#0|f|A5|off|off|on#1", "A4|off|on|off#1|g|A3|on|off|off#2"),
            List.of("A4|off|on|off#1|g|A3|on|off|off#2", "A5|off|off|on#1|h|A3|on|off|off#1"),
            List.of("A5|off|off|on#1|h|A3|on|off|off#1", "A3|on|off|off#2|e|A4|off|on|off#2", "A3|on|off|off#2|f|A5|off|off|on#2"),
            List.of("A3|on|off|off#2|e|A4|off|on|off#2", "A3|on|off|off#2|f|A5|off|off|on#2", "A3|on|off|off#1|e|A4|off|on|off#1", "A3|on|off|off#1|f|A5|off|off|on#1"),
            List.of("A3|on|off|off#2|f|A5|off|off|on#2", "A3|on|off|off#1|e|A4|off|on|off#1", "A3|on|off|off#1|f|A5|off|off|on#1", "A4|off|on|off#2|g|A3|on|off|off#2"),
            List.of("A3|on|off|off#1|e|A4|off|on|off#1", "A3|on|off|off#1|f|A5|off|off|on#1", "A4|off|on|off#2|g|A3|on|off|off#2", "A5|off|off|on#2|h|A3|on|off|off#0"),
            List.of("A3|on|off|off#1|f|A5|off|off|on#1", "A4|off|on|off#2|g|A3|on|off|off#2", "A5|off|off|on#2|h|A3|on|off|off#0")
        ));
    }

    private static Case caseGR1Test16() {
        return new Case("GR1Test16", "fsp/Blocking/ControllableFSPs/GR1Test16.lts", null, List.of(
            List.of("A0|off#0|a|A1|off#0", "A0|off#0|u|A5|off#0"),
            List.of("A0|off#0|u|A5|off#0", "A1|off#0|b|A2|off#0", "A1|off#0|d|A3|off#0"),
            List.of("A1|off#0|b|A2|off#0", "A1|off#0|d|A3|off#0", "A5|off#0|f|A2|off#0"),
            List.of("A1|off#0|d|A3|off#0", "A5|off#0|f|A2|off#0", "A2|off#0|c|A1|off#0"),
            List.of("A5|off#0|f|A2|off#0", "A2|off#0|c|A1|off#0", "A3|off#0|e|A4|on#0"),
            List.of("A2|off#0|c|A1|off#0", "A3|off#0|e|A4|on#0"),
            List.of("A3|off#0|e|A4|on#0"),
            List.of("A4|on#0|g|A3|off#0")
        ));
    }

    private static Case caseGR1Test22() {
        return new Case("GR1Test22", "fsp/Blocking/ControllableFSPs/GR1Test22.lts", null, List.of(
            List.of("A0|off_off#0|a|A1|off_off#0", "A0|off_off#0|b|A3|off_off#0"),
            List.of("A0|off_off#0|b|A3|off_off#0", "A1|off_off#0|c|A2|on_off#0"),
            List.of("A1|off_off#0|c|A2|on_off#0", "A3|off_off#0|e|A4|off_on#0"),
            List.of("A3|off_off#0|e|A4|off_on#0", "A2|on_off#0|d|A1|off_off#0"),
            List.of("A2|on_off#0|d|A1|off_off#0", "A4|off_on#0|f|A3|off_off#0"),
            List.of("A4|off_on#0|f|A3|off_off#0")
        ));
    }

    private static Case caseGR1Test24() {
        return new Case("GR1Test24", "fsp/Blocking/ControllableFSPs/GR1Test24.lts", null, List.of(
            List.of("A1|off#0|a|A2|off#0"),
            List.of("A2|off#0|b|A3|on#0", "A2|off#0|c|A4|off#0", "A2|off#0|d|A5|on#0"),
            List.of("A2|off#0|c|A4|off#0", "A2|off#0|d|A5|on#0", "A3|on#0|e|A3|on#0"),
            List.of("A2|off#0|d|A5|on#0", "A3|on#0|e|A3|on#0", "A4|off#0|f|A2|off#0"),
            List.of("A3|on#0|e|A3|on#0", "A4|off#0|f|A2|off#0", "A5|on#0|g|A5|on#0"),
            List.of("A4|off#0|f|A2|off#0", "A5|on#0|g|A5|on#0"),
            List.of("A5|on#0|g|A5|on#0")
        ));
    }

    // ── NO CONTROLLABLE ───────────────────────────────────────────────────────

    private static Case caseGR1Test4() {
        return new Case("GR1Test4", "fsp/Blocking/NoControllableFSPs/GR1Test4.lts", null, List.of(
            List.of("A0|off|off#0|a|A1|on|off#0"),
            List.of("A1|on|off#0|b|A2|off|off#1", "A1|on|off#0|c|A3|off|off#1"),
            List.of("A1|on|off#0|c|A3|off|off#1", "A2|off|off#1|d|A0|off|on#1"),
            List.of("A2|off|off#1|d|A0|off|on#1", "A3|off|off#1|e|A0|off|off#1"),
            List.of("A3|off|off#1|e|A0|off|off#1", "A0|off|on#1|a|A1|on|off#0"),
            List.of("A0|off|on#1|a|A1|on|off#0", "A0|off|off#1|a|A1|on|off#1"),
            List.of("A0|off|off#1|a|A1|on|off#1"),
            List.of("A1|on|off#1|b|A2|off|off#1", "A1|on|off#1|c|A3|off|off#1"),
            List.of("A1|on|off#1|c|A3|off|off#1")
        ));
    }

    private static Case caseGR1Test6() {
        return new Case("GR1Test6", "fsp/Blocking/NoControllableFSPs/GR1Test6.lts", null, List.of(
            List.of("A0|off|off#0|a|A1|off|off#0"),
            List.of("A1|off|off#0|b|A2|off|off#0", "A1|off|off#0|c|A3|off|off#0"),
            List.of("A1|off|off#0|c|A3|off|off#0", "A2|off|off#0|d|A0|off|on#0"),
            List.of("A2|off|off#0|d|A0|off|on#0", "A3|off|off#0|e|A0|on|off#0"),
            List.of("A3|off|off#0|e|A0|on|off#0", "A0|off|on#0|a|A1|off|off#0"),
            List.of("A0|off|on#0|a|A1|off|off#0", "A0|on|off#0|a|A1|off|off#1"),
            List.of("A0|on|off#0|a|A1|off|off#1"),
            List.of("A1|off|off#1|b|A2|off|off#1", "A1|off|off#1|c|A3|off|off#1"),
            List.of("A1|off|off#1|c|A3|off|off#1", "A2|off|off#1|d|A0|off|on#1"),
            List.of("A2|off|off#1|d|A0|off|on#1", "A3|off|off#1|e|A0|on|off#1"),
            List.of("A3|off|off#1|e|A0|on|off#1", "A0|off|on#1|a|A1|off|off#0"),
            List.of("A0|off|on#1|a|A1|off|off#0", "A0|on|off#1|a|A1|off|off#1")
        ));
    }

    private static Case caseGR1Test11() {
        return new Case("GR1Test11", "fsp/Blocking/NoControllableFSPs/GR1Test11.lts", null, List.of(
            List.of("A0|off|off#0|a|A1|off|off#0", "A0|off|off#0|c|A2|off|off#0"),
            List.of("A0|off|off#0|c|A2|off|off#0", "A1|off|off#0|b|A0|on|off#0"),
            List.of("A1|off|off#0|b|A0|on|off#0", "A2|off|off#0|d|A0|off|on#0"),
            List.of("A2|off|off#0|d|A0|off|on#0", "A0|on|off#0|a|A1|off|off#1", "A0|on|off#0|c|A2|off|off#1"),
            List.of("A0|on|off#0|a|A1|off|off#1", "A0|on|off#0|c|A2|off|off#1", "A0|off|on#0|a|A1|off|off#0", "A0|off|on#0|c|A2|off|off#0"),
            List.of("A0|on|off#0|c|A2|off|off#1", "A0|off|on#0|a|A1|off|off#0", "A0|off|on#0|c|A2|off|off#0", "A1|off|off#1|b|A0|on|off#1"),
            List.of("A0|off|on#0|a|A1|off|off#0", "A0|off|on#0|c|A2|off|off#0", "A1|off|off#1|b|A0|on|off#1", "A2|off|off#1|d|A0|off|on#1"),
            List.of("A0|off|on#0|c|A2|off|off#0", "A1|off|off#1|b|A0|on|off#1", "A2|off|off#1|d|A0|off|on#1"),
            List.of("A1|off|off#1|b|A0|on|off#1", "A2|off|off#1|d|A0|off|on#1"),
            List.of("A2|off|off#1|d|A0|off|on#1", "A0|on|off#1|a|A1|off|off#1", "A0|on|off#1|c|A2|off|off#1"),
            List.of("A0|on|off#1|a|A1|off|off#1", "A0|on|off#1|c|A2|off|off#1", "A0|off|on#1|a|A1|off|off#0", "A0|off|on#1|c|A2|off|off#0"),
            List.of("A0|on|off#1|c|A2|off|off#1", "A0|off|on#1|a|A1|off|off#0", "A0|off|on#1|c|A2|off|off#0")
        ));
    }

    private static Case caseGR1Test12() {
        return new Case("GR1Test12", "fsp/Blocking/NoControllableFSPs/GR1Test12.lts", null, List.of(
            List.of("A0|off|off#0|a|A1|off|off#0", "A0|off|off#0|c|A2|off|off#0"),
            List.of("A0|off|off#0|c|A2|off|off#0", "A1|off|off#0|b|A0|on|off#0"),
            List.of("A1|off|off#0|b|A0|on|off#0", "A2|off|off#0|d|A0|off|on#0"),
            List.of("A2|off|off#0|d|A0|off|on#0", "A0|on|off#0|a|A1|off|off#1", "A0|on|off#0|c|A2|off|off#1"),
            List.of("A0|on|off#0|a|A1|off|off#1", "A0|on|off#0|c|A2|off|off#1", "A0|off|on#0|a|A1|off|off#0", "A0|off|on#0|c|A2|off|off#0"),
            List.of("A0|on|off#0|c|A2|off|off#1", "A0|off|on#0|a|A1|off|off#0", "A0|off|on#0|c|A2|off|off#0", "A1|off|off#1|b|A0|on|off#1"),
            List.of("A0|off|on#0|a|A1|off|off#0", "A0|off|on#0|c|A2|off|off#0", "A1|off|off#1|b|A0|on|off#1", "A2|off|off#1|d|A0|off|on#1"),
            List.of("A0|off|on#0|c|A2|off|off#0", "A1|off|off#1|b|A0|on|off#1", "A2|off|off#1|d|A0|off|on#1"),
            List.of("A1|off|off#1|b|A0|on|off#1", "A2|off|off#1|d|A0|off|on#1"),
            List.of("A2|off|off#1|d|A0|off|on#1", "A0|on|off#1|a|A1|off|off#1", "A0|on|off#1|c|A2|off|off#1"),
            List.of("A0|on|off#1|a|A1|off|off#1", "A0|on|off#1|c|A2|off|off#1", "A0|off|on#1|a|A1|off|off#0", "A0|off|on#1|c|A2|off|off#0"),
            List.of("A0|on|off#1|c|A2|off|off#1", "A0|off|on#1|a|A1|off|off#0", "A0|off|on#1|c|A2|off|off#0")
        ));
    }

    private static Case caseGR1Test17() {
        return new Case("GR1Test17", "fsp/Blocking/NoControllableFSPs/GR1Test17.lts", null, List.of(
            List.of("A0|off|off#0|d|A1|off|off#0", "A0|off|off#0|e|A2|off|off#0"),
            List.of("A0|off|off#0|e|A2|off|off#0", "A1|off|off#0|c|A0|off|off#0"),
            List.of("A1|off|off#0|c|A0|off|off#0", "A2|off|off#0|a|A3|on|off#0", "A2|off|off#0|b|A3|off|on#0"),
            List.of("A2|off|off#0|a|A3|on|off#0", "A2|off|off#0|b|A3|off|on#0"),
            List.of("A2|off|off#0|b|A3|off|on#0", "A3|on|off#0|f|A0|off|off#1"),
            List.of("A3|on|off#0|f|A0|off|off#1", "A3|off|on#0|f|A0|off|off#0"),
            List.of("A3|off|on#0|f|A0|off|off#0", "A0|off|off#1|d|A1|off|off#1", "A0|off|off#1|e|A2|off|off#1"),
            List.of("A0|off|off#1|d|A1|off|off#1", "A0|off|off#1|e|A2|off|off#1"),
            List.of("A0|off|off#1|e|A2|off|off#1", "A1|off|off#1|c|A0|off|off#1"),
            List.of("A1|off|off#1|c|A0|off|off#1", "A2|off|off#1|a|A3|on|off#1", "A2|off|off#1|b|A3|off|on#1"),
            List.of("A2|off|off#1|a|A3|on|off#1", "A2|off|off#1|b|A3|off|on#1"),
            List.of("A2|off|off#1|b|A3|off|on#1", "A3|on|off#1|f|A0|off|off#1")
        ));
    }

    private static Case caseGR1Test18() {
        return new Case("GR1Test18", "fsp/Blocking/NoControllableFSPs/GR1Test18.lts", null, List.of(
            List.of("A0|off_off#0|a|A1|on_off#0", "A0|off_off#0|b|A2|off_off#0"),
            List.of("A0|off_off#0|b|A2|off_off#0", "A1|on_off#0|d|A4|off_off#0"),
            List.of("A1|on_off#0|d|A4|off_off#0", "A2|off_off#0|c|A3|off_on#0", "A2|off_off#0|z|A5|off_off#0"),
            List.of("A2|off_off#0|c|A3|off_on#0", "A2|off_off#0|z|A5|off_off#0", "A4|off_off#0|f|A0|off_off#0"),
            List.of("A2|off_off#0|z|A5|off_off#0", "A4|off_off#0|f|A0|off_off#0", "A3|off_on#0|e|A4|off_off#0"),
            List.of("A4|off_off#0|f|A0|off_off#0", "A3|off_on#0|e|A4|off_off#0", "A5|off_off#0|g|A5|off_off#0"),
            List.of("A3|off_on#0|e|A4|off_off#0", "A5|off_off#0|g|A5|off_off#0"),
            List.of("A5|off_off#0|g|A5|off_off#0")
        ));
    }

    private static Case caseGR1Test19() {
        return new Case("GR1Test19", "fsp/Blocking/NoControllableFSPs/GR1Test19.lts", null, List.of(
            List.of("A0|off#0|a|A1|off#0"),
            List.of("A1|off#0|b|A2|off#0", "A1|off#0|d|A3|off#0"),
            List.of("A1|off#0|d|A3|off#0", "A2|off#0|c|A1|off#0", "A2|off#0|e|E|off#0"),
            List.of("A2|off#0|c|A1|off#0", "A2|off#0|e|E|off#0", "A3|off#0|f|A4|off#0"),
            List.of("A2|off#0|e|E|off#0", "A3|off#0|f|A4|off#0"),
            List.of("A3|off#0|f|A4|off#0"),
            List.of("A4|off#0|g|G|on#0", "A4|off#0|u|A1|off#0"),
            List.of("A4|off#0|u|A1|off#0", "G|on#0|g|G|on#0"),
            List.of("G|on#0|g|G|on#0")
        ));
    }

    private static Case caseGR1Test20() {
        return new Case("GR1Test20", "fsp/Blocking/NoControllableFSPs/GR1Test20.lts", null, List.of(
            List.of("A1|off#0|b|A2|off#0", "A1|off#0|a|A3|off#0"),
            List.of("A1|off#0|a|A3|off#0", "A2|off#0|c|A3|off#0", "A2|off#0|e|A4|off#0"),
            List.of("A2|off#0|c|A3|off#0", "A2|off#0|e|A4|off#0", "A3|off#0|d|A2|off#0", "A3|off#0|g|A5|off#0"),
            List.of("A2|off#0|e|A4|off#0", "A3|off#0|d|A2|off#0", "A3|off#0|g|A5|off#0"),
            List.of("A3|off#0|d|A2|off#0", "A3|off#0|g|A5|off#0", "A4|off#0|f|A2|off#0", "A4|off#0|i|A5|off#0", "A4|off#0|l|A7|off#0"),
            List.of("A3|off#0|g|A5|off#0", "A4|off#0|f|A2|off#0", "A4|off#0|i|A5|off#0", "A4|off#0|l|A7|off#0"),
            List.of("A4|off#0|f|A2|off#0", "A4|off#0|i|A5|off#0", "A4|off#0|l|A7|off#0", "A5|off#0|h|A3|off#0", "A5|off#0|j|A6|off#0", "A5|off#0|o|A8|on#0"),
            List.of("A4|off#0|i|A5|off#0", "A4|off#0|l|A7|off#0", "A5|off#0|h|A3|off#0", "A5|off#0|j|A6|off#0", "A5|off#0|o|A8|on#0"),
            List.of("A4|off#0|l|A7|off#0", "A5|off#0|h|A3|off#0", "A5|off#0|j|A6|off#0", "A5|off#0|o|A8|on#0"),
            List.of("A5|off#0|h|A3|off#0", "A5|off#0|j|A6|off#0", "A5|off#0|o|A8|on#0", "A7|off#0|m|A7|off#0", "A7|off#0|n|A2|off#0"),
            List.of("A5|off#0|j|A6|off#0", "A5|off#0|o|A8|on#0", "A7|off#0|m|A7|off#0", "A7|off#0|n|A2|off#0"),
            List.of("A5|off#0|o|A8|on#0", "A7|off#0|m|A7|off#0", "A7|off#0|n|A2|off#0", "A6|off#0|k|A4|off#0")
        ));
    }

    private static Case caseGR1Test21() {
        return new Case("GR1Test21", "fsp/Blocking/NoControllableFSPs/GR1Test21.lts", null, List.of(
            List.of("A1|off#0|b|A2|off#0", "A1|off#0|a|A3|off#0"),
            List.of("A1|off#0|a|A3|off#0", "A2|off#0|c|A3|off#0", "A2|off#0|e|A4|off#0"),
            List.of("A2|off#0|c|A3|off#0", "A2|off#0|e|A4|off#0", "A3|off#0|d|A2|off#0", "A3|off#0|g|A5|off#0"),
            List.of("A2|off#0|e|A4|off#0", "A3|off#0|d|A2|off#0", "A3|off#0|g|A5|off#0"),
            List.of("A3|off#0|d|A2|off#0", "A3|off#0|g|A5|off#0", "A4|off#0|f|A2|off#0", "A4|off#0|i|A5|off#0", "A4|off#0|l|A7|off#0"),
            List.of("A3|off#0|g|A5|off#0", "A4|off#0|f|A2|off#0", "A4|off#0|i|A5|off#0", "A4|off#0|l|A7|off#0"),
            List.of("A4|off#0|f|A2|off#0", "A4|off#0|i|A5|off#0", "A4|off#0|l|A7|off#0", "A5|off#0|h|A3|off#0", "A5|off#0|j|A6|off#0", "A5|off#0|o|A8|on#0"),
            List.of("A4|off#0|i|A5|off#0", "A4|off#0|l|A7|off#0", "A5|off#0|h|A3|off#0", "A5|off#0|j|A6|off#0", "A5|off#0|o|A8|on#0"),
            List.of("A4|off#0|l|A7|off#0", "A5|off#0|h|A3|off#0", "A5|off#0|j|A6|off#0", "A5|off#0|o|A8|on#0"),
            List.of("A5|off#0|h|A3|off#0", "A5|off#0|j|A6|off#0", "A5|off#0|o|A8|on#0", "A7|off#0|m|A7|off#0", "A7|off#0|n|A2|off#0"),
            List.of("A5|off#0|j|A6|off#0", "A5|off#0|o|A8|on#0", "A7|off#0|m|A7|off#0", "A7|off#0|n|A2|off#0"),
            List.of("A5|off#0|o|A8|on#0", "A7|off#0|m|A7|off#0", "A7|off#0|n|A2|off#0", "A6|off#0|k|A4|off#0")
        ));
    }

    private static Case caseGR1Test23() {
        return new Case("GR1Test23", "fsp/Blocking/NoControllableFSPs/GR1Test23.lts", null, List.of(
            List.of("A1|off#0|a|A2|off#0"),
            List.of("A2|off#0|b|A3|on#0", "A2|off#0|c|A4|off#0", "A2|off#0|d|A5|on#0"),
            List.of("A2|off#0|c|A4|off#0", "A2|off#0|d|A5|on#0", "A3|on#0|e|A3|on#0"),
            List.of("A2|off#0|d|A5|on#0", "A3|on#0|e|A3|on#0", "A4|off#0|f|A2|off#0"),
            List.of("A3|on#0|e|A3|on#0", "A4|off#0|f|A2|off#0", "A5|on#0|g|A5|off#0"),
            List.of("A4|off#0|f|A2|off#0", "A5|on#0|g|A5|off#0"),
            List.of("A5|on#0|g|A5|off#0"),
            List.of("A5|off#0|g|A5|off#0")
        ));
    }

    // ── TL-2-2 ───────────────────────────────────────────────────────────────

    private static Case caseTL22() {
        return new Case("TL-2-2", "fsp/Blocking/Benchmark/TL/TL-2-2.fsp",
            new int[]{0, 3, 6, 9, 12, 14, 17, 18, 14, 15, 22, 25, 27, 22, 28, 17},
            List.of(
                List.of(
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[0]|ERROR|Testing|off#0"
                ),
                List.of(
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[2]|Operative[0]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|put[1]|Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[1]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[1]|Operative[0]|Working[0]|ERROR|Testing|off#0"
                ),
                List.of(
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[2]|Operative[0]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[1]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[1]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[1]|Working[0]|ERROR|Testing|off#0"
                ),
                List.of(
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[2]|Operative[0]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[1]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[1]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[1]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[2]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|put[2]|Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[1]|ERROR|Testing|off#0"
                ),
                List.of(
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[2]|Operative[0]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[1]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[1]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[1]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[2]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[1]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0"
                ),
                List.of(
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[2]|Operative[0]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[1]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[1]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[1]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[2]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[1]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[0]|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|return[1]|Working[0]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|accept|Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0"
                ),
                List.of(
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[2]|Operative[0]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[1]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[1]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[1]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[2]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[1]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[0]|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|accept|Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0|get[1]|Working[0]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0|reject|Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|on#0"
                ),
                List.of(
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[2]|Operative[0]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[1]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[1]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[1]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[2]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[1]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[0]|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|accept|Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0|get[1]|Working[0]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|on#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|on#0|get[1]|Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|on#0|get[2]|Working[0]|Operative[1]|Working[0]|ERROR|Testing|off#0"
                ),
                List.of(
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[2]|Operative[0]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[1]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[1]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[1]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[2]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[1]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[0]|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|accept|Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0|get[1]|Working[0]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|on#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|on#0|get[2]|Working[0]|Operative[1]|Working[0]|ERROR|Testing|off#0"
                ),
                List.of(
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[2]|Operative[0]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[1]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[1]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[1]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[2]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[1]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[0]|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|accept|Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|on#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|on#0|get[2]|Working[0]|Operative[1]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0|get[2]|Working[0]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0|get[0]|Working[1]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0|get[1]|Working[0]|ERROR|Working[2]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0|put[2]|Working[0]|Operative[0]|Working[0]|Operative[1]|_Testing_0|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0|reject|Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|on#0"
                ),
                List.of(
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[2]|Operative[0]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[1]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[1]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[1]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[2]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[1]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[0]|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|accept|Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|on#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|on#0|get[2]|Working[0]|Operative[1]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0|get[2]|Working[0]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0|get[0]|Working[1]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0|get[1]|Working[0]|ERROR|Working[2]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0|reject|Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|on#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|_Testing_0|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[1]|_Testing_0|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|_Testing_0|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[1]|_Testing_0|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|_Testing_0|off#0|reject|Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|on#0"
                ),
                List.of(
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[2]|Operative[0]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[1]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[1]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[1]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[2]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[1]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[0]|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Testing|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|on#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|on#0|get[2]|Working[0]|Operative[1]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0|get[2]|Working[0]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0|get[0]|Working[1]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0|get[1]|Working[0]|ERROR|Working[2]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0|reject|Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|on#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|_Testing_0|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[1]|_Testing_0|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|_Testing_0|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[1]|_Testing_0|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|on#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|on#0|get[1]|Working[0]|ERROR|Working[1]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|on#0|get[2]|Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0"
                ),
                List.of(
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[2]|Operative[0]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[1]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[1]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[1]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[2]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[1]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[0]|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Testing|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|on#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|on#0|get[2]|Working[0]|Operative[1]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0|get[2]|Working[0]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0|get[0]|Working[1]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0|get[1]|Working[0]|ERROR|Working[2]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|_Testing_0|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[1]|_Testing_0|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|_Testing_0|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[1]|_Testing_0|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|on#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|on#0|get[1]|Working[0]|ERROR|Working[1]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|on#0|get[0]|Working[1]|Operative[0]|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|on#0|get[1]|Working[0]|ERROR|Working[2]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|on#0|put[2]|Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|on#0|get[2]|Working[0]|Operative[0]|Working[1]|ERROR|Testing|off#0"
                ),
                List.of(
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[2]|Operative[0]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[1]|Working[1]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[1]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[1]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[1]|Working[0]|ERROR|Working[2]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|off#0|get[2]|Working[0]|Operative[0]|Working[1]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[0]|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Testing|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Testing|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|on#0|get[0]|Working[1]|Operative[1]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[1]|Working[0]|Operative[0]|Idle|on#0|get[2]|Working[0]|Operative[1]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0|get[1]|Working[0]|ERROR|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[0]|Idle|on#0|get[2]|Working[0]|Operative[0]|Working[0]|ERROR|Testing|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0|get[0]|Working[1]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|_Testing_0|off#0|get[1]|Working[0]|ERROR|Working[2]|Operative[0]|_Testing_0|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|_Testing_0|off#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[1]|_Testing_0|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|_Testing_0|off#0|get[1]|Working[0]|ERROR|Working[1]|Operative[1]|_Testing_0|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|on#0|get[0]|Working[1]|Operative[0]|Working[0]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[0]|Operative[1]|Idle|on#0|get[1]|Working[0]|ERROR|Working[1]|Operative[1]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|on#0|get[0]|Working[1]|Operative[0]|Working[1]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|on#0|get[1]|Working[0]|ERROR|Working[2]|Operative[0]|Idle|off#0",
                    "Working[0]|Operative[0]|Working[1]|Operative[0]|Idle|on#0|get[2]|Working[0]|Operative[0]|Working[1]|ERROR|Testing|off#0"
                )
            )
        );
    }
}
