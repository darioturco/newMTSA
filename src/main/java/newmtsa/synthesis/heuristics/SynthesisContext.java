package newmtsa.synthesis.heuristics;

import newmtsa.parser.ast.LTS;
import newmtsa.synthesis.ExtendedTransition;

import java.util.List;
import java.util.Set;

/**
 * Read-only view of the OTF-DCS exploration state, provided to context-aware
 * heuristics via {@link Heuristic#init(SynthesisContext)}.
 *
 * <p>All returned collections are <em>live views</em> — they reflect the
 * current state of the synthesis engine at the time of each call.
 */
public interface SynthesisContext {

    /**
     * Full component list used in the parallel composition
     * (plant components followed by safety monitors).
     */
    List<LTS> components();

    /**
     * Pre-computed marked states for each component, index-aligned with
     * {@link #components()}.  An empty set means "all states are marked"
     * (FSP convention for processes without explicit marking).
     */
    List<Set<String>> componentMarked();

    /** Set of controllable event labels. */
    Set<String> controllable();

    /**
     * Composite states that have been expanded so far
     * (the key set of the exploration structure).
     */
    Set<String> exploredStates();

    /** Composite states confirmed as winning (W⁺). */
    Set<String> goals();

    /**
     * Outgoing transitions of the given composite state in the exploration
     * structure.  Returns an empty list if the state has not been expanded yet.
     */
    List<ExtendedTransition> successorsOf(String compositeState);

    /**
     * Returns the "plant-state key" for a composite state — the portion that
     * can be split on {@code "|"} to yield per-component sub-states.
     *
     * <p>The default implementation returns the state unchanged (non-blocking DCS).
     * GR(1) overrides this to strip the phase suffix
     * (e.g. {@code "c0|c1#2"} → {@code "c0|c1"}).
     */
    default String plantStateKey(String compositeState) {
        return compositeState;
    }
}
