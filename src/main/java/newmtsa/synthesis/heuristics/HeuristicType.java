package newmtsa.synthesis.heuristics;

import newmtsa.synthesis.heuristics.mcts.MCTSHeuristic;
import newmtsa.synthesis.heuristics.ra.RAHeuristic;

/**
 * Catalogue of available exploration heuristics.
 *
 * <p>Use {@link #create()} to obtain a fresh instance of the chosen strategy.
 *
 * <pre>
 * FIRST         – always picks the first transition in the frontier (deterministic, fast)
 * RANDOM        – picks uniformly at random
 * BFS           – breadth-first layer-by-layer expansion
 * HUMAN         – interactive: prints the frontier and asks the user to choose
 * RA            – Ready Abstraction (corrected formulation, Pazos 2024)
 * RA_R          – RA + recompute estimates when new marked states are discovered
 * RA_E          – RA + structure-aware tie-breaking (best standalone improvement)
 * RA_ER         – RA.R + RA.E combined (best overall)
 * RA_ERG        – RA.R + RA.E + Goals-as-targets (all improvements)
 *
 * Open-queue variants (+_OPEN suffix): restrict heuristic picks to "open" states
 * (states with no remaining pending uncontrollable transitions), implementing a DFS
 * bias. RA_OPEN beats plain RA; combining open queue with RA.E/RA.ER/RA.ERG is
 * counterproductive per Pazos (2024) §7.2 — provided for completeness only.
 * RA_OPEN       – RA  + open queue
 * RA_R_OPEN     – RA.R + open queue
 * RA_E_OPEN     – RA.E + open queue
 * RA_ER_OPEN    – RA.ER + open queue
 * RA_ERG_OPEN   – RA.ERG + open queue
 *
 * RL            – Marker for Python-driven Reinforcement Learning training.
 *                 Selection is overridden externally via DCSForPython.expand(idx);
 *                 features (BASIC/ROL) computed by FeatureCompute. Falls back
 *                 to FIRST when the engine runs autonomously.
 * MCTS_RL       – MCTS guided by an ONNX policy/value model (autonomous run).
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
    RA_ERG,
    RA_OPEN,
    RA_R_OPEN,
    RA_E_OPEN,
    RA_ER_OPEN,
    RA_ERG_OPEN,
    /**
     * Marker for Python-driven RL training.
     * Frontier picks come from the Python agent via {@code DCSForPython.expand(idx)};
     * this Java heuristic is bypassed and only used as a fallback (picks first).
     * Pair with a {@code FeatureType} (BASIC / ROL) at the {@code DCSForPython} level.
     */
    RL,
    /**
     * MCTS guided by a flat (MLP) ONNX model.
     * Model path: JVM system property {@code mcts.onnx.path} (default: {@code model.onnx}).
     * Simulation count / cPuct / depth: {@code mcts.simulations}, {@code mcts.cpuct}, {@code mcts.depth}.
     */
    MCTS_RL;

    /** Creates and returns a fresh {@link Heuristic} instance for this type. */
    public Heuristic create() {
        return switch (this) {
            case FIRST         -> new FirstSelectionHeuristic();
            case RANDOM        -> new RandomHeuristic();
            case BFS           -> new BFSHeuristic();
            case HUMAN         -> new HumanHeuristic();
            case RA            -> RAHeuristic.base();
            case RA_R          -> RAHeuristic.withR();
            case RA_E          -> RAHeuristic.withE();
            case RA_ER         -> RAHeuristic.withER();
            case RA_ERG        -> RAHeuristic.withERG();
            case RA_OPEN       -> RAHeuristic.base().withOpenQueue();
            case RA_R_OPEN     -> RAHeuristic.withR().withOpenQueue();
            case RA_E_OPEN     -> RAHeuristic.withE().withOpenQueue();
            case RA_ER_OPEN    -> RAHeuristic.withER().withOpenQueue();
            case RA_ERG_OPEN   -> RAHeuristic.withERG().withOpenQueue();
            case RL            -> new RLHeuristic(
                    System.getProperty("rl.onnx.path"),
                    System.getProperty("feature_type", "ROL"));
            case MCTS_RL       -> new MCTSHeuristic(
                    System.getProperty("mcts.onnx.path", "model.onnx"),
                    System.getProperty("feature_type", "ROL"),
                    Integer.parseInt(System.getProperty("mcts.simulations", "50")),
                    Double.parseDouble(System.getProperty("mcts.cpuct",       "1.5")),
                    Integer.parseInt(System.getProperty("mcts.depth",        "10")));
        };
    }
}
