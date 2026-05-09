package newmtsa.synthesis.heuristics.mcts;

import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.heuristics.SynthesisContext;

import java.util.*;

/**
 * Immutable snapshot of a virtual synthesis frontier used during MCTS rollouts.
 *
 * <p>Each {@link #expand} call returns a new VirtualState without mutating {@code this},
 * so MCTS can branch freely without undoing state.
 *
 * <p>Virtual successor computation for unexplored states uses the component LTS
 * transition tables ({@code compTrans}, {@code compAlpha}) reconstructed once in
 * {@link MCTSHeuristic#init}.  For states already explored by the real engine,
 * {@link SynthesisContext#successorsOf} is used directly (O(1) hash lookup).
 */
class VirtualState {

    final Set<String>              explored;   // real explored ∪ virtually expanded
    final List<ExtendedTransition> frontier;   // current virtual pending list
    final String                   lastFrom;
    final String                   lastTo;

    VirtualState(Set<String>              explored,
                 List<ExtendedTransition> frontier,
                 String                   lastFrom,
                 String                   lastTo) {
        this.explored = explored;
        this.frontier = frontier;
        this.lastFrom = lastFrom;
        this.lastTo   = lastTo;
    }

    /**
     * Expand transition {@code t} and return the resulting virtual state.
     *
     * @param t            transition to expand
     * @param ctx          live synthesis context (goals, errors, successorsOf)
     * @param compTrans    per-component transition tables: action → (from → to)
     * @param compAlpha    per-component alphabets
     * @param succCache    mutable cache for computed virtual successors; cleared each real step
     */
    VirtualState expand(ExtendedTransition             t,
                        SynthesisContext               ctx,
                        List<Map<String, Map<String, String>>> compTrans,
                        List<Set<String>>              compAlpha,
                        Map<String, List<ExtendedTransition>> succCache) {

        // Build new frontier without t
        List<ExtendedTransition> newFrontier = new ArrayList<>(frontier);
        newFrontier.remove(t);

        String to = t.to();

        // Dead or winning branch — no children to add
        if (ctx.errors().contains(to) || ctx.goals().contains(to)) {
            return new VirtualState(explored, newFrontier, t.from(), to);
        }

        // Already virtually explored — don't re-expand
        if (explored.contains(to)) {
            return new VirtualState(explored, newFrontier, t.from(), to);
        }

        Set<String> newExplored = new HashSet<>(explored);
        newExplored.add(to);

        // Get successors: real engine first, then compute from component LTS
        List<ExtendedTransition> succs = ctx.successorsOf(to);
        if (succs.isEmpty()) {
            succs = succCache.computeIfAbsent(to, s -> computeSuccessors(s, compTrans, compAlpha));
        }

        Set<String> goals  = ctx.goals();
        Set<String> errors = ctx.errors();
        for (ExtendedTransition s : succs) {
            if (!newExplored.contains(s.to()) && !goals.contains(s.to()) && !errors.contains(s.to())) {
                newFrontier.add(s);
            }
        }

        return new VirtualState(newExplored, newFrontier, t.from(), to);
    }

    /**
     * Compute successors of {@code state} from the component LTS product.
     * Mirrors the logic in {@code OTFDirectedControledSyntesisNonBlocking.exploreState()}.
     */
    private static List<ExtendedTransition> computeSuccessors(
            String                                state,
            List<Map<String, Map<String, String>>> compTrans,
            List<Set<String>>                     compAlpha) {

        int      n     = compTrans.size();
        String[] parts = state.split("\\|", n);
        if (parts.length < n) parts = Arrays.copyOf(parts, n);

        // Collect all enabled actions across components
        Set<String> actions = new LinkedHashSet<>();
        for (int i = 0; i < n; i++) actions.addAll(compTrans.get(i).keySet());

        List<ExtendedTransition> result = new ArrayList<>();
        for (String action : actions) {
            String[] next   = parts.clone();
            boolean  enabled = true;
            for (int i = 0; i < n; i++) {
                if (!compAlpha.get(i).contains(action)) continue;
                Map<String, String> tr = compTrans.get(i).get(action);
                if (tr == null || !tr.containsKey(parts[i])) { enabled = false; break; }
                String dest = tr.get(parts[i]);
                if ("ERROR".equals(dest)) { enabled = false; break; }
                next[i] = dest;
            }
            if (enabled) {
                result.add(new ExtendedTransition(state, action, String.join("|", next)));
            }
        }
        return result;
    }
}
