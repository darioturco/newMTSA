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
}
