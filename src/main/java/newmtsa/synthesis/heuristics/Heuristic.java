package newmtsa.synthesis.heuristics;

import newmtsa.synthesis.ExtendedTransition;

import java.util.List;

/** Strategy for picking the next transition to explore from the pending frontier. */
public interface Heuristic {

    /**
     * Pick one transition from the pending list.
     * The returned transition must be present in {@code pending}.
     */
    ExtendedTransition pick(List<ExtendedTransition> pending);

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
}
