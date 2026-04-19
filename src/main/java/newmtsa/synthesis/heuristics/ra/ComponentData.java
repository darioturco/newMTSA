package newmtsa.synthesis.heuristics.ra;

import newmtsa.parser.ast.LTS;
import newmtsa.parser.ast.Transition;

import java.util.*;

/**
 * Pre-computed, immutable per-component data for the RA heuristic.
 *
 * <p>Built once per component at {@link RAHeuristic} initialisation.  The most
 * expensive field is {@link #distToEachMarked}, which is populated via a
 * single reverse-BFS per marked state — {@code O(|S_j| × |A_j|)} overall.
 */
final class ComponentData {

    /** Events in this component's alphabet. */
    final Set<String> alphabet;

    /**
     * Marked states for this component.
     * An empty set means this component has no marking constraint.
     */
    final Set<String> markedStates;

    /** {@code action → (from-state → to-state)} lookup. */
    final Map<String, Map<String, String>> trans;

    /** {@code state → set of events enabled at that state}. */
    final Map<String, Set<String>> enabledAt;

    /**
     * Shortest-path distance from any state to each marked state, computed via
     * reverse BFS.
     *
     * <p>Layout: {@code markedState → (anyState → distance)}.
     * If a state does not appear as a key in the inner map, it cannot reach
     * that marked state.
     */
    final Map<String, Map<String, Integer>> distToEachMarked;

    // ── constructor ───────────────────────────────────────────────────────────

    /**
     * @param lts              the component automaton
     * @param preComputedMarked marked states for this component as pre-computed
     *                          by the synthesis engine (may differ from
     *                          {@code lts.acceptingStates()} when marking actions
     *                          are used, or empty for safety monitors)
     */
    ComponentData(LTS lts, Set<String> preComputedMarked) {

        // ── alphabet, trans, enabledAt ────────────────────────────────────────
        Set<String>                      alpha   = new LinkedHashSet<>();
        Map<String, Map<String, String>> tMap    = new HashMap<>();
        Map<String, Set<String>>         enabled = new HashMap<>();

        for (Transition tr : lts.transitions()) {
            alpha.add(tr.action());
            tMap.computeIfAbsent(tr.action(), k -> new HashMap<>())
                .put(tr.from(), tr.to());
            enabled.computeIfAbsent(tr.from(), k -> new LinkedHashSet<>())
                   .add(tr.action());
        }

        this.alphabet      = Collections.unmodifiableSet(alpha);
        this.trans         = Collections.unmodifiableMap(tMap);
        this.enabledAt     = Collections.unmodifiableMap(enabled);
        this.markedStates  = Collections.unmodifiableSet(
                                 new LinkedHashSet<>(preComputedMarked));

        // ── BFS distances to each marked state ────────────────────────────────
        // For each marked state m, reverse-BFS from m gives d(s, m) for all s
        // that can reach m in the forward direction.
        Map<String, Map<String, Integer>> dists = new HashMap<>();

        if (!markedStates.isEmpty()) {
            // Build reverse adjacency once, shared across all BFS runs.
            Map<String, List<String>> rev = new HashMap<>();
            for (Transition tr : lts.transitions()) {
                rev.computeIfAbsent(tr.to(), k -> new ArrayList<>())
                   .add(tr.from());
            }

            for (String m : markedStates) {
                Map<String, Integer> dist  = new HashMap<>();
                Queue<String>        queue = new ArrayDeque<>();
                dist.put(m, 0);
                queue.add(m);
                while (!queue.isEmpty()) {
                    String s = queue.poll();
                    int    d = dist.get(s);
                    for (String pred : rev.getOrDefault(s, List.of())) {
                        if (!dist.containsKey(pred)) {
                            dist.put(pred, d + 1);
                            queue.add(pred);
                        }
                    }
                }
                dists.put(m, dist);
            }
        }

        this.distToEachMarked = Collections.unmodifiableMap(dists);
    }
}
