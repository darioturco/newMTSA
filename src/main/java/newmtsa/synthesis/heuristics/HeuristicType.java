package newmtsa.synthesis.heuristics;

import newmtsa.synthesis.heuristics.ra.RAHeuristic;

/**
 * Catalogue of available exploration heuristics.
 *
 * <p>Use {@link #create()} to obtain a fresh instance of the chosen strategy.
 *
 * <pre>
 * FIRST    – always picks the first transition in the frontier (deterministic, fast)
 * RANDOM   – picks uniformly at random
 * BFS      – breadth-first layer-by-layer expansion
 * HUMAN    – interactive: prints the frontier and asks the user to choose
 * RA       – Ready Abstraction (corrected formulation, Pazos 2024)
 * RA_R     – RA + recompute estimates when new marked states are discovered
 * RA_E     – RA + structure-aware tie-breaking (best standalone improvement)
 * RA_ER    – RA.R + RA.E combined (best overall)
 * RA_ERG   – RA.R + RA.E + Goals-as-targets (all improvements)
 * </pre>
 */
public enum HeuristicType {

    FIRST,
    RANDOM,
    BFS,
    HUMAN,
    RA,
    RA_R,
    RA_E,
    RA_ER,
    RA_ERG;

    /** Creates and returns a fresh {@link Heuristic} instance for this type. */
    public Heuristic create() {
        return switch (this) {
            case FIRST   -> new FirstSelectionHeuristic();
            case RANDOM  -> new RandomHeuristic();
            case BFS     -> new BFSHeuristic();
            case HUMAN   -> new HumanHeuristic();
            case RA      -> RAHeuristic.base();
            case RA_R    -> RAHeuristic.withR();
            case RA_E    -> RAHeuristic.withE();
            case RA_ER   -> RAHeuristic.withER();
            case RA_ERG  -> RAHeuristic.withERG();
        };
    }
}
