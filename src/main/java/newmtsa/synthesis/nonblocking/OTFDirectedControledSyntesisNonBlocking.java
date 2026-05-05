package newmtsa.synthesis.nonblocking;

import newmtsa.parser.ast.LTS;
import newmtsa.parser.ast.LtlPropertyDef;
import newmtsa.synthesis.DCSForPython;
import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.SynthesisResult;
import newmtsa.synthesis.heuristics.Heuristic;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * On-the-fly Directed Controller Synthesis for safe and non-blocking properties.
 *
 * <p>Implements the otf-dcs algorithm from:
 * <blockquote>
 *   Ciolek et al., "On-the-fly informed search of non-blocking directed controllers",
 *   Automatica 147 (2023) 110731.
 * </blockquote>
 *
 * <p>The algorithm incrementally explores the parallel composition of the plant
 * automata one transition at a time, guided by a heuristic. States are classified
 * into three sets that are kept as an invariant throughout exploration:
 * <ul>
 *   <li><b>Goals</b> (W<sup>⊕</sup>): states from which the controller can guarantee
 *       reaching a marked state without blocking.</li>
 *   <li><b>Errors</b> (W<sup>⊖</sup>): states from which no such controller exists.</li>
 *   <li><b>None</b>: not yet classified.</li>
 * </ul>
 *
 * <p><b>Winning condition (non-blocking)</b>: the controlled plant must never reach a
 * deadlock that is not a marked state, and from every reachable state there must exist
 * a finite continuation to some marked state.
 *
 * <p><b>State representation</b>: a composite state is the string
 * {@code "s0|s1|...|sk"} joining the current state of each component LTS.
 *
 * <p><b>Marked states</b>: a composite state is marked iff, for every component
 * whose {@code acceptingStates} set is non-empty, the component's current sub-state
 * is in that set.  Components with an empty {@code acceptingStates} set are treated
 * as having all states marked (FSP convention for processes without explicit marking).
 *
 * <p>All algorithm state is held by {@link #dcsForPython}; this class is a thin
 * driver that applies the heuristic inside {@link #run()}.
 */
public class OTFDirectedControledSyntesisNonBlocking {

    // ── core engine ───────────────────────────────────────────────────────────

    /** The step-by-step DCS engine (also accessible from Python via JPype). */
    public final DCSForPython dcsForPython;

    // ── run options ───────────────────────────────────────────────────────────

    private final boolean verbose;
    private final int     expansionLimit;
    private final boolean useNumericIds;

    // ── constructors ──────────────────────────────────────────────────────────

    /**
     * @param components       automata that form the plant (parallel composition)
     * @param safetyProperties LTL safety properties (optional, may be empty)
     * @param markingActions   event labels that, when fired, lead to a marked state
     * @param controllable     set of controllable event labels
     * @param heuristic        frontier selection strategy
     * @param verbose          print trace to stdout when true
     */
    public OTFDirectedControledSyntesisNonBlocking(List<LTS>            components,
                                                   List<LtlPropertyDef> safetyProperties,
                                                   Set<String>          markingActions,
                                                   Set<String>          controllable,
                                                   Heuristic            heuristic,
                                                   boolean              verbose) {
        this(components, safetyProperties, markingActions, controllable,
             heuristic, verbose, Integer.MAX_VALUE, false);
    }

    /**
     * @param expansionLimit   stop and return UNREALIZABLE after this many
     *                         transitions expanded (use {@link Integer#MAX_VALUE}
     *                         for unlimited)
     */
    public OTFDirectedControledSyntesisNonBlocking(List<LTS>            components,
                                                   List<LtlPropertyDef> safetyProperties,
                                                   Set<String>          markingActions,
                                                   Set<String>          controllable,
                                                   Heuristic            heuristic,
                                                   boolean              verbose,
                                                   int                  expansionLimit) {
        this(components, safetyProperties, markingActions, controllable,
             heuristic, verbose, expansionLimit, false);
    }

    /**
     * @param useNumericIds    when true, verbose output prints state integer IDs
     *                         instead of state names
     */
    public OTFDirectedControledSyntesisNonBlocking(List<LTS>            components,
                                                   List<LtlPropertyDef> safetyProperties,
                                                   Set<String>          markingActions,
                                                   Set<String>          controllable,
                                                   Heuristic            heuristic,
                                                   boolean              verbose,
                                                   int                  expansionLimit,
                                                   boolean              useNumericIds) {
        this.verbose        = verbose;
        this.expansionLimit = expansionLimit;
        this.useNumericIds  = useNumericIds;
        this.dcsForPython   = new DCSForPython("", components, safetyProperties,
                                               markingActions, controllable, heuristic);
    }

    // ── public entry point ────────────────────────────────────────────────────

    /**
     * Runs the otf-dcs algorithm (Listing 1 of Ciolek et al. 2023).
     *
     * @return {@link SynthesisResult#isRealizable()} true iff a non-blocking
     *         director exists from the initial state.
     */
    public SynthesisResult run() {
        log("Initial state explored | states=" + dcsForPython.getStatesExplored());

        if (dcsForPython.isExplorationEnded()) {
            log("Initial state is losing (deadlock or illegal) — UNREALIZABLE");
            return dcsForPython.getSynthesisResult();
        }

        while (!dcsForPython.isExplorationEnded()) {
            List<ExtendedTransition> frontier = dcsForPython.getFrontier();
            if (frontier.isEmpty()) break;

            logFrontier(frontier);

            int index = dcsForPython.heuristic.pick(frontier);
            if (index < 0 || index >= frontier.size()) {
                System.err.println("[DCS-NB] WARNING: heuristic returned invalid index " + index
                        + " for frontier of size " + frontier.size() + " — using nearest valid index");
                index = Math.max(0, Math.min(index, frontier.size() - 1));
            }
            ExtendedTransition t = frontier.get(index);

            log("  step " + (dcsForPython.getTransitionsExplored() + 1)
                    + " | " + index + " | " + formatState(t.from()) + " --[" + t.action() + "]--> " + formatState(t.to()));

            dcsForPython.expand(index);

            if (dcsForPython.isRealizable()) {
                log("s0 ∈ Goals — REALIZABLE"
                        + " | states=" + dcsForPython.getStatesExplored()
                        + " transitions=" + dcsForPython.getTransitionsExplored());
                return dcsForPython.getSynthesisResult();
            }
            if (dcsForPython.isExplorationEnded()) {
                log("s0 ∈ Errors — UNREALIZABLE"
                        + " | states=" + dcsForPython.getStatesExplored()
                        + " transitions=" + dcsForPython.getTransitionsExplored());
                return dcsForPython.getSynthesisResult();
            }
            if (dcsForPython.getTransitionsExplored() >= expansionLimit) {
                log("Budget exhausted (" + expansionLimit + ") — aborting");
                return SynthesisResult.unrealizable(dcsForPython.getStatesExplored(),
                                                    dcsForPython.getTransitionsExplored());
            }
        }

        log("Exploration complete"
                + " | states=" + dcsForPython.getStatesExplored()
                + " transitions=" + dcsForPython.getTransitionsExplored());
        return dcsForPython.getSynthesisResult();
    }

    // ── verbose helpers ───────────────────────────────────────────────────────

    private void log(String msg) {
        if (verbose) System.out.println("[DCS-NB] " + msg);
    }

    private void logFrontier(List<ExtendedTransition> frontier) {
        if (!verbose) return;
        System.out.println("[DCS-NB] frontier (" + frontier.size() + "):");
        for (int i = 0; i < frontier.size(); i++) {
            ExtendedTransition ft = frontier.get(i);
            System.out.println("        " + i + " | " + formatState(ft.from())
                    + " --[" + ft.action() + "]--> " + formatState(ft.to()));
        }
    }

    /**
     * Returns the display string for a composite state.
     * When {@link #useNumericIds} is true, each component sub-state is replaced
     * by its integer ID from {@link newmtsa.parser.ast.LTS#stateIndex()}.
     */
    private String formatState(String composite) {
        if (!useNumericIds) return composite;
        java.util.List<newmtsa.parser.ast.LTS> comps = dcsForPython.getComponents();
        String[] parts = composite.split("\\|", comps.size());
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < parts.length; i++) {
            if (i > 0) sb.append('|');
            Integer id = (i < comps.size()) ? comps.get(i).stateIndex().get(parts[i]) : null;
            sb.append(id != null ? id : parts[i]);
        }
        return sb.toString();
    }
}
