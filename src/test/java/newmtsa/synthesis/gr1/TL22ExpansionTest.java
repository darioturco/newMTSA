package newmtsa.synthesis.gr1;

import newmtsa.synthesis.DCSForPython;
import newmtsa.synthesis.heuristics.HeuristicType;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks that RA heuristic on TL-2-2 does not expand more transitions
 * than the reference original implementation (16 expansions).
 *
 * The original GR1 RA uses a state-ranked priority queue (CompostateRanker).
 * Our implementation must match that ordering via source-state comparison in
 * compareTransitions.
 */
class TL22ExpansionTest {

    private static final String PATH = "fsp/Blocking/Benchmark/TL/TL-2-2.fsp";
    private static final int    ORIGINAL_EXPANSIONS = 16;

    @ParameterizedTest(name = "{0}")
    @EnumSource(value = HeuristicType.class, names = {"RA", "RA_R", "RA_E", "RA_ER", "RA_ERG"})
    void tl22_ra_expansions_match_original(HeuristicType type) throws Exception {
        DCSForPython dcs = DCSForPython.fromPath(PATH, type.name(), "");
        dcs.gr1Engine.run();
        int expansions = dcs.gr1Engine.getTransitionsExplored();
        assertTrue(expansions <= ORIGINAL_EXPANSIONS,
                type + " expanded " + expansions + " transitions, expected <= " + ORIGINAL_EXPANSIONS);
    }
}
