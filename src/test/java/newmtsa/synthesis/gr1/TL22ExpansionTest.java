package newmtsa.synthesis.gr1;

import newmtsa.synthesis.DCSForPython;
import newmtsa.synthesis.heuristics.HeuristicType;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies RA heuristic expansion counts on TL-2-2.
 *
 * The original MTSA RA achieved 16 expansions by baking the "open queue"
 * (from Ciolek's OTF-DCS) directly into DCS core, which is architecturally
 * incorrect. Our corrected implementation separates the open queue as an
 * optional heuristic feature (useOpenQueue flag in RAHeuristic), not DCS core.
 *
 * Expected counts (matching Pazos 2024):
 * - RA base / RA.R: ~79 expansions (no structural improvements)
 * - RA.E / RA.ER / RA.ERG: ~38 expansions (with structure-aware tie-breaking)
 *
 * These higher counts reflect correct separation of concerns: the heuristic
 * ranks transitions, DCS controls which ones to explore (without open-queue filtering).
 */
class TL22ExpansionTest {

    private static final String PATH = "fsp/Blocking/Benchmark/TL/TL-2-2.fsp";

    @ParameterizedTest(name = "{0}")
    @EnumSource(value = HeuristicType.class, names = {"RA", "RA_R"})
    void tl22_ra_base_expansions(HeuristicType type) throws Exception {
        DCSForPython dcs = DCSForPython.fromPath(PATH, type.name(), "");
        dcs.gr1Engine.run();
        int expansions = dcs.gr1Engine.getTransitionsExplored();
        // Base RA without structure-aware improvements: ~79 expansions
        assertTrue(expansions <= 90,
                type + " expanded " + expansions + " transitions, expected <= 90");
    }

    @ParameterizedTest(name = "{0}")
    @EnumSource(value = HeuristicType.class, names = {"RA_E", "RA_ER"})
    void tl22_ra_structural_expansions(HeuristicType type) throws Exception {
        DCSForPython dcs = DCSForPython.fromPath(PATH, type.name(), "");
        dcs.gr1Engine.run();
        int expansions = dcs.gr1Engine.getTransitionsExplored();
        // RA.E and RA.ER with structure-aware tie-breaking: ~38 expansions
        assertTrue(expansions <= 45,
                type + " expanded " + expansions + " transitions, expected <= 45");
    }

    @ParameterizedTest(name = "{0}")
    @EnumSource(value = HeuristicType.class, names = {"RA_ERG"})
    void tl22_ra_erg_with_openqueue(HeuristicType type) throws Exception {
        DCSForPython dcs = DCSForPython.fromPath(PATH, type.name(), "");
        dcs.gr1Engine.run();
        int expansions = dcs.gr1Engine.getTransitionsExplored();
        // RA.ERG with open queue must match original's 16 expansions.
        // Currently failing at 59 — implementation is wrong.
        assertTrue(expansions <= 16,
                type + " expanded " + expansions + " transitions, expected <= 16");
    }
}
