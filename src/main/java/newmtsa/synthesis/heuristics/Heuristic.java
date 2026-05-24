package newmtsa.synthesis.heuristics;

import newmtsa.synthesis.Director;
import newmtsa.synthesis.ExtendedTransition;
import java.util.List;

/** Strategy for picking the next transition to explore from the pending frontier. */
public interface Heuristic {

    /**
     * Pick one transition from the pending list and return its 0-based index.
     * The returned value must be in [0, pending.size()).
     */
    int pick(List<ExtendedTransition> pending);

    /**
     * Called once by the synthesis engine after it is fully initialised,
     * before the first call to {@link #pick}.
     *
     * <p>Context-aware heuristics (e.g. the RA family) override this to obtain
     * a live view of exploration state.  The default implementation is a no-op,
     * so existing stateless heuristics need not change.
     *
     * @param ctx live, read-only view of the synthesis engine's internal state
     */
    default void init(SynthesisContext ctx) {}

    /**
     * Print frontier state and transition scores before expansion (if verbose enabled).
     * Heuristics should override to show relevant scoring info (e.g., RA shows distances).
     * Called by the synthesis engine after a transition is picked but before expansion.
     *
     * @param pending current frontier
     * @param pickedIndex index of the transition that was picked
     */
    default void printFrontier(List<ExtendedTransition> pending, int pickedIndex) {}

    /**
     * Called by the synthesis engine once, immediately before it returns the final result.
     * Heuristics that accumulate statistics or decisions during exploration should override
     * this to print or persist their summary.
     *
     * @param result the final {@link Director} (may be unrealizable, in which case
     *               {@link Director#enabled()} is empty)
     */
    default void notifyExplorationEnd(Director result) {}
}
