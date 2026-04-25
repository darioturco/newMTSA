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

        // ── reverse adjacency (shared by distToEachMarked and stepsToEnableAction) ──
        Map<String, List<String>> rev = new HashMap<>();
        for (Transition tr : lts.transitions()) {
            rev.computeIfAbsent(tr.to(), k -> new ArrayList<>()).add(tr.from());
        }

        // ── BFS distances to each marked state ────────────────────────────────
        // For each marked state m, reverse-BFS from m gives d(s, m) for all s
        // that can reach m in the forward direction.
        Map<String, Map<String, Integer>> dists = new HashMap<>();

        if (!markedStates.isEmpty()) {
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

        // ── reachableFromState ────────────────────────────────────────────────
        // Forward BFS from every state: used by RAHeuristic to check whether
        // action l can transitively enable action t.
        Map<String, Set<String>> reach = new HashMap<>();
        for (String start : enabled.keySet()) {
            Set<String>   visited = new LinkedHashSet<>();
            Queue<String> q       = new ArrayDeque<>();
            q.add(start); visited.add(start);
            while (!q.isEmpty()) {
                String cur = q.poll();
                for (Map.Entry<String, Map<String, String>> e : tMap.entrySet()) {
                    String next = e.getValue().get(cur);
                    if (next != null && !visited.contains(next)) {
                        visited.add(next); q.add(next);
                    }
                }
            }
            reach.put(start, Collections.unmodifiableSet(visited));
        }
        this.reachableFromState = Collections.unmodifiableMap(reach);

        // ── stepsToEnableAction ───────────────────────────────────────────────
        // For each action t: reverse-BFS from all states where t is enabled
        // gives the minimum steps from any state to reach a state where t fires.
        Map<String, Map<String, Integer>> ste = new HashMap<>();
        for (String act : alpha) {
            // Collect states where 'act' is enabled.
            Set<String> enabledStates = new HashSet<>();
            for (Map.Entry<String, Set<String>> e : enabled.entrySet()) {
                if (e.getValue().contains(act)) enabledStates.add(e.getKey());
            }
            if (enabledStates.isEmpty()) { ste.put(act, Map.of()); continue; }

            // Reverse-BFS from all enabled states simultaneously.
            Map<String, Integer> dist  = new HashMap<>();
            Queue<String>        q     = new ArrayDeque<>();
            for (String s : enabledStates) { dist.put(s, 0); q.add(s); }
            while (!q.isEmpty()) {
                String s = q.poll();
                int    d = dist.get(s);
                for (String pred : rev.getOrDefault(s, List.of())) {
                    if (!dist.containsKey(pred)) {
                        dist.put(pred, d + 1); q.add(pred);
                    }
                }
            }
            ste.put(act, Collections.unmodifiableMap(dist));
        }
        this.stepsToEnableAction = Collections.unmodifiableMap(ste);
    }

    /** All states reachable (including itself) from each state. */
    final Map<String, Set<String>> reachableFromState;

    /**
     * For each action t: minimum steps from any state s to reach a state where
     * t is enabled.  Layout: action → (state → steps).  A missing inner entry
     * means t is unreachable from that state.
     */
    final Map<String, Map<String, Integer>> stepsToEnableAction;
}
