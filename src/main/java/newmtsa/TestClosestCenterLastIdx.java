package newmtsa;

import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.features.FeaturesContext;
import newmtsa.synthesis.features.SuperGeneralFeatures;

import java.util.*;

/**
 * Throwaway: verifies is_closest_center_last_idx_in_action for CM-like actions.
 * Delete after use.
 *
 * CM action: mouse[i][move[j]]  — mouse i moves to position j.
 * Safe place for CM-n-k = k (center of [0..2k]).
 * Candidates: all mice moving to various positions.
 * Feature should fire for the candidate(s) whose destination j is closest to center.
 */
public class TestClosestCenterLastIdx {

    public static void main(String[] args) {
        // superdfs.family = CM → full mode (not slim)
        System.setProperty("superdfs.family", "CM");

        // Minimal FeaturesContext with 3 components (simulating CM-3-1: 3 mice, positions 0..2)
        FeaturesContext ctx = new FeaturesContext(
            Collections.emptySet(),  // goals
            Collections.emptySet(),  // errors
            Collections.emptySet(),  // none
            Collections.emptyMap(),  // succMap
            Collections.emptyMap(),  // parents
            Set.of("mouse[0][move[0]]", "mouse[0][move[1]]", "mouse[0][move[2]]",
                   "mouse[1][move[0]]", "mouse[1][move[1]]", "mouse[1][move[2]]",
                   "mouse[2][move[0]]", "mouse[2][move[1]]", "mouse[2][move[2]]"),  // controllable
            Collections.emptySet(),  // alphabet
            List.of(),               // components (empty ok — only used for roles/marked)
            List.of()                // compMarked
        );

        SuperGeneralFeatures feat = new SuperGeneralFeatures();
        feat.init(ctx);

        // ── Test 1: CM-3-1 (k=1, positions 0..2, safe=1) ─────────────────────
        // Candidates: mouse[0][move[0]], mouse[1][move[1]], mouse[2][move[2]]
        // Destinations: 0, 1, 2  →  center = 1  →  closest = mouse[1][move[1]]
        List<ExtendedTransition> cands1 = List.of(
            new ExtendedTransition("A", "mouse[0][move[0]]", "B"),
            new ExtendedTransition("A", "mouse[1][move[1]]", "C"),
            new ExtendedTransition("A", "mouse[2][move[2]]", "D")
        );
        feat.precompute(cands1);
        checkFeature("Test1 mouse[0][move[0]] (dest=0, expect 0)", feat, cands1.get(0), 0);
        checkFeature("Test1 mouse[1][move[1]] (dest=1, expect 1)", feat, cands1.get(1), 1);
        checkFeature("Test1 mouse[2][move[2]] (dest=2, expect 0)", feat, cands1.get(2), 0);

        // ── Test 2: ties — two candidates equidistant from center ─────────────
        // Destinations: 0, 2  →  center = 1  →  both equidistant (dist=1 each)
        List<ExtendedTransition> cands2 = List.of(
            new ExtendedTransition("A", "mouse[0][move[0]]", "B"),
            new ExtendedTransition("A", "mouse[1][move[2]]", "C")
        );
        feat.precompute(cands2);
        checkFeature("Test2 mouse[0][move[0]] (dest=0, dist=1, expect 1 — tied)", feat, cands2.get(0), 1);
        checkFeature("Test2 mouse[1][move[2]] (dest=2, dist=1, expect 1 — tied)", feat, cands2.get(1), 1);

        // ── Test 3: single candidate — always fires ────────────────────────────
        List<ExtendedTransition> cands3 = List.of(
            new ExtendedTransition("A", "mouse[0][move[2]]", "B")
        );
        feat.precompute(cands3);
        checkFeature("Test3 single candidate (expect 1)", feat, cands3.get(0), 1);

        // ── Test 4: action without last index — feature must stay 0 ──────────
        List<ExtendedTransition> cands4 = List.of(
            new ExtendedTransition("A", "approve", "B"),
            new ExtendedTransition("A", "refuse",  "C")
        );
        feat.precompute(cands4);
        checkFeature("Test4 approve (no bracket, expect 0)", feat, cands4.get(0), 0);

        System.out.println("\nDone.");
    }

    // Feature index found dynamically via names.indexOf — works in both slim and full mode.
    private static void checkFeature(String label, SuperGeneralFeatures feat,
                                     ExtendedTransition t, int expected) {
        float[] f     = feat.compute(t);
        List<String> names = feat.getFeatureNames();
        int idx = names.indexOf("is_closest_center_last_idx_in_action");
        int actual = (int) f[idx];
        String status = actual == expected ? "OK" : "FAIL";
        System.out.printf("[%s] %s  idx=%d  got=%d  expected=%d%n",
            status, label, idx, actual, expected);
    }
}
