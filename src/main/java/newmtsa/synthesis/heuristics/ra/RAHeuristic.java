package newmtsa.synthesis.heuristics.ra;

import newmtsa.parser.ast.LTS;
import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.heuristics.Heuristic;
import newmtsa.synthesis.heuristics.SynthesisContext;

import java.util.*;

/**
 * Ready Abstraction (RA) heuristic for non-blocking OTF-DCS.
 *
 * <p>For each candidate transition {@code (e, ℓ, e')} on the frontier, RA
 * estimates how promising it is by analysing the modular component structure
 * independently — avoiding the state explosion of the full composition.
 *
 * <p>For each component {@code E_j} that has a non-empty marking set, the
 * heuristic computes a tuple {@code ⟨m_j, d_j⟩} where {@code m_j} encodes
 * confidence in reaching a marked state ({@code -1}=goal, {@code 0}=visited
 * marked, {@code 1}=unvisited marked, {@code 2}=unreachable) and {@code d_j}
 * is the estimated number of composite steps.  Transitions are ranked by
 * comparing their tuple sequences lexicographically after sorting them
 * descending (Definition 9 of Pazos 2024).
 *
 * <h2>Variants</h2>
 * <dl>
 *   <dt><b>RA</b> (base)</dt>
 *   <dd>Plain Ready Abstraction with the corrected formulation from Pazos 2024.</dd>
 *
 *   <dt><b>RA.R</b></dt>
 *   <dd>Recomputes stale estimates whenever new marked states are discovered
 *       during exploration, at the cost of polynomial-time cache invalidation.</dd>
 *
 *   <dt><b>RA.E</b></dt>
 *   <dd>Structure-aware tie-breaking: controllable transitions are further
 *       prioritised by (1) preferring source states with no uncontrollable
 *       successors ("controllable states"), then (2) preferring targets not
 *       yet explored.  Largest performance gain for symmetric problems.</dd>
 *
 *   <dt><b>RA.G</b></dt>
 *   <dd>Includes states in the Goals set as high-priority targets ({@code m=-1})
 *       in the estimate, biasing exploration toward already-winning states.</dd>
 * </dl>
 *
 * <p>Use the static factory methods to select a variant:
 * {@link #base()}, {@link #withR()}, {@link #withE()}, {@link #withER()},
 * {@link #withERG()}.
 */
public class RAHeuristic implements Heuristic {

    // ── configuration ─────────────────────────────────────────────────────────
    private final boolean useR;   // RA.R: recompute estimates on new marked-state discoveries
    private final boolean useE;   // RA.E: structure-aware tie-breaking
    private final boolean useG;   // RA.G: use Goals as additional targets

    // ── per-component data (built in init()) ──────────────────────────────────
    private List<ComponentData> compData;
    private Set<String>         controllable;
    private int                 numComponents;

    // ── live exploration state (from context) ─────────────────────────────────
    private SynthesisContext ctx;

    /**
     * For each component j: E_j sub-states that have appeared in any explored
     * composite state.  Refreshed lazily when the explored-states count grows.
     */
    private List<Set<String>> visitedCompStates;

    /**
     * For each component j: E_j sub-states that appear in any Goals composite
     * state.  Used only when {@link #useG} is true.
     */
    private List<Set<String>> goalCompStates;

    // ── estimate cache ────────────────────────────────────────────────────────
    /** {@code compositeState → action → estimate tuple list}. */
    private final Map<String, Map<String, List<EstimateTuple>>> estimateCache
            = new HashMap<>();

    private int lastExploredCount      = 0;
    private int lastVisitedMarkedCount = 0;
    private int lastGoalsCount         = 0;

    // ── private constructor + static factories ────────────────────────────────

    private RAHeuristic(boolean useR, boolean useE, boolean useG) {
        this.useR = useR;
        this.useE = useE;
        this.useG = useG;
    }

    /** Plain RA — corrected formulation. */
    public static RAHeuristic base()    { return new RAHeuristic(false, false, false); }
    /** RA.R — recompute estimates when new marked states are discovered. */
    public static RAHeuristic withR()   { return new RAHeuristic(true,  false, false); }
    /** RA.E — structure-aware tie-breaking. */
    public static RAHeuristic withE()   { return new RAHeuristic(false, true,  false); }
    /** RA.ER — recompute + structure-aware (best single-improvement combination). */
    public static RAHeuristic withER()  { return new RAHeuristic(true,  true,  false); }
    /** RA.ERG — all improvements enabled. */
    public static RAHeuristic withERG() { return new RAHeuristic(true,  true,  true);  }

    // ── Heuristic lifecycle ───────────────────────────────────────────────────

    /**
     * Called by the synthesis engine once all internal state is ready.
     * Builds per-component data from the context and prepares tracking sets.
     */
    @Override
    public void init(SynthesisContext ctx) {
        this.ctx          = ctx;
        this.controllable = ctx.controllable();

        List<LTS>         comps  = ctx.components();
        List<Set<String>> marked = ctx.componentMarked();
        int n = comps.size();
        this.numComponents     = n;
        this.compData          = new ArrayList<>(n);
        this.visitedCompStates = new ArrayList<>(n);
        this.goalCompStates    = new ArrayList<>(n);

        for (int j = 0; j < n; j++) {
            compData.add(new ComponentData(comps.get(j), marked.get(j)));
            visitedCompStates.add(new HashSet<>());
            goalCompStates.add(new HashSet<>());
        }
    }

    // ── Heuristic.pick ────────────────────────────────────────────────────────

    @Override
    public ExtendedTransition pick(List<ExtendedTransition> pending) {
        if (ctx == null) return pending.get(0);  // not yet initialised — safe fallback

        refreshVisitedStates();
        if (useG) refreshGoalStates();

        // Linear scan: find the transition that is "smallest" under the RA ordering.
        ExtendedTransition best = null;
        for (ExtendedTransition t : pending) {
            if (best == null || compareTransitions(t, best) < 0) {
                best = t;
            }
        }
        return best;
    }

    // ── lazy state refresh ────────────────────────────────────────────────────

    /**
     * Rebuilds {@link #visitedCompStates} whenever new composite states have
     * been explored.  Also invalidates the estimate cache when RA.R is active
     * and new marked states have been reached.
     */
    private void refreshVisitedStates() {
        Set<String> explored = ctx.exploredStates();
        int current = explored.size();
        if (current == lastExploredCount) return;

        for (Set<String> set : visitedCompStates) set.clear();

        int markedCount = 0;
        for (String composite : explored) {
            String[] parts = splitCompositeState(composite);
            for (int j = 0; j < numComponents && j < parts.length; j++) {
                visitedCompStates.get(j).add(parts[j]);
            }
            if (isMarkedComposite(parts)) markedCount++;
        }

        // RA.R: clear stale estimates when new marked states appear.
        if (useR && markedCount > lastVisitedMarkedCount) {
            estimateCache.clear();
        }
        lastVisitedMarkedCount = markedCount;
        lastExploredCount      = current;
    }

    /**
     * Rebuilds {@link #goalCompStates} whenever the Goals set has grown
     * (only called when {@link #useG} is true).  Clears the estimate cache
     * because {@code mFlag} values depend on goal component states.
     */
    private void refreshGoalStates() {
        Set<String> goals = ctx.goals();
        int current = goals.size();
        if (current == lastGoalsCount) return;

        for (Set<String> set : goalCompStates) set.clear();
        for (String composite : goals) {
            String[] parts = splitCompositeState(composite);
            for (int j = 0; j < numComponents && j < parts.length; j++) {
                goalCompStates.get(j).add(parts[j]);
            }
        }

        estimateCache.clear();   // goal projections changed → cached m-flags stale
        lastGoalsCount = current;
    }

    // ── comparison ────────────────────────────────────────────────────────────

    /**
     * Returns negative if transition {@code a} should be expanded before {@code b}.
     *
     * <p>Ordering rules (in priority order):
     * <ol>
     *   <li>Uncontrollable actions before controllable (environment moves are mandatory).</li>
     *   <li>RA.E structural priority (controllable only): controllable-source states first,
     *       then unexplored targets.</li>
     *   <li>RA estimate: tuple sequences sorted descending, then compared lexicographically.
     *       For controllable transitions, smaller = better (closer to goal).
     *       For uncontrollable, larger = better (classify hard states early).</li>
     * </ol>
     */
    private int compareTransitions(ExtendedTransition a, ExtendedTransition b) {
        boolean aCtrl = controllable.contains(a.action());
        boolean bCtrl = controllable.contains(b.action());

        if (!aCtrl &&  bCtrl) return -1;   // a uncontrollable, b controllable → a first
        if ( aCtrl && !bCtrl) return +1;   // a controllable, b uncontrollable → b first

        // RA.E: structural tie-breaking for controllable transitions.
        if (useE && aCtrl) {
            int sp = compareStructural(a, b);
            if (sp != 0) return sp;
        }

        // RA estimate comparison.
        List<EstimateTuple> estA = getOrComputeEstimate(a);
        List<EstimateTuple> estB = getOrComputeEstimate(b);

        // Sort each tuple sequence descending before lexicographic comparison (Def. 9).
        List<EstimateTuple> keyA = sortedDescending(estA);
        List<EstimateTuple> keyB = sortedDescending(estB);
        int lex = compareLex(keyA, keyB);

        // Controllable: ascending (smaller estimate = closer to goal = preferred).
        // Uncontrollable: descending (larger estimate = harder state to classify first).
        return aCtrl ? lex : -lex;
    }

    // ── RA.E structural comparison ────────────────────────────────────────────

    /**
     * Structural comparison for RA.E: controllable-source states come first,
     * then source states with no explored controllable children yet.
     *
     * <p>Matches the reference implementation's {@code CompostateRanker}:
     * rule 1 = {@code uncontrollablesCount == 0},
     * rule 2 = {@code getControllablesExpandedCount() == 0}.
     */
    private int compareStructural(ExtendedTransition a, ExtendedTransition b) {
        // Rule 1: source is a "controllable state" (no uncontrollable outgoing transitions).
        boolean aCtrlSrc = isControllableState(a.from());
        boolean bCtrlSrc = isControllableState(b.from());
        if ( aCtrlSrc && !bCtrlSrc) return -1;
        if (!aCtrlSrc &&  bCtrlSrc) return +1;

        // Rule 2: source has no explored controllable children yet — fresh source first.
        boolean aFresh = noExploredControllableChildren(a.from());
        boolean bFresh = noExploredControllableChildren(b.from());
        if ( aFresh && !bFresh) return -1;
        if (!aFresh &&  bFresh) return +1;

        return 0;
    }

    /**
     * Returns true iff no controllable transition from {@code from} has a
     * visited target — i.e., no controllable child of this source state has
     * been expanded yet.  Mirrors {@code Compostate.getControllablesExpandedCount() == 0}.
     */
    private boolean noExploredControllableChildren(String from) {
        for (ExtendedTransition t : ctx.successorsOf(from)) {
            if (controllable.contains(t.action()) && ctx.exploredStates().contains(t.to())) {
                return false;
            }
        }
        return true;
    }

    /**
     * A composite state is "controllable" if every transition in its explored
     * successor list is controllable (the environment has no forced moves).
     * States that have not been expanded yet are treated as controllable
     * (empty successor list → no uncontrollable transitions known).
     */
    private boolean isControllableState(String compositeState) {
        for (ExtendedTransition t : ctx.successorsOf(compositeState)) {
            if (!controllable.contains(t.action())) return false;
        }
        return true;
    }

    // ── estimate computation ──────────────────────────────────────────────────

    /** Returns the cached estimate, computing it on first access. */
    private List<EstimateTuple> getOrComputeEstimate(ExtendedTransition t) {
        return estimateCache
                .computeIfAbsent(t.from(), k -> new HashMap<>())
                .computeIfAbsent(t.action(),
                                 k -> computeEstimate(t.from(), t.action()));
    }

    /**
     * Computes the full estimate for {@code action} from composite state
     * {@code compositeState}: one {@link EstimateTuple} per component with a
     * non-empty marking set.
     */
    private List<EstimateTuple> computeEstimate(String compositeState, String action) {
        String[]            parts  = splitCompositeState(compositeState);
        List<EstimateTuple> result = new ArrayList<>();

        for (int j = 0; j < numComponents; j++) {
            if (compData.get(j).markedStates.isEmpty()) continue;
            String e_j = (j < parts.length) ? parts[j] : null;
            if (e_j == null) {
                result.add(EstimateTuple.UNREACHABLE);
                continue;
            }
            result.add(computeComponentEstimate(j, e_j, action));
        }
        return result;
    }

    /**
     * Per-component estimate for {@code action} from sub-state {@code e_j} in
     * component {@code j}.
     *
     * <p>Case tree (follows Definition 8 of Pazos 2024):
     * <ol>
     *   <li>Action fires in E_j and changes state → BFS distance from successor
     *       to nearest marked state.</li>
     *   <li>Action is a self-loop or absent from A_j (E_j stays at e_j):
     *     <ul>
     *       <li>If e_j itself is marked → {@code ⟨m, 1⟩}.</li>
     *       <li>Otherwise scan ready events {@code ℓ''} at e_j that make
     *           progress; return {@code ⟨m, 1 + dist(succ_j'', marked)⟩}
     *           for the best such event.</li>
     *     </ul>
     *   </li>
     * </ol>
     */
    private EstimateTuple computeComponentEstimate(int j, String e_j, String action) {
        ComponentData cd = compData.get(j);

        String  e_j_succ           = cd.trans.getOrDefault(action, Map.of()).get(e_j);
        boolean isSelfloopOrAbsent = (e_j_succ == null || e_j_succ.equals(e_j));

        if (!isSelfloopOrAbsent) {
            // Case 1: action fires and advances E_j.
            return bestDistanceToMarked(j, e_j_succ);
        }

        // Cases 2+: E_j stays at e_j after firing action.
        if (cd.markedStates.contains(e_j)) {
            // e_j is already a marked state; one composite step is enough.
            return new EstimateTuple(mFlag(j, e_j), 1);
        }

        // Look for any ready event ℓ'' at e_j that advances E_j toward a marked state.
        EstimateTuple best = EstimateTuple.UNREACHABLE;
        for (String alt : cd.enabledAt.getOrDefault(e_j, Set.of())) {
            if (alt.equals(action)) continue;
            String alt_succ = cd.trans.getOrDefault(alt, Map.of()).get(e_j);
            if (alt_succ == null || alt_succ.equals(e_j)) continue;  // skip self-loops

            EstimateTuple reached = bestDistanceToMarked(j, alt_succ);
            if (reached == EstimateTuple.UNREACHABLE) continue;

            // +1 for the composite step that fires ℓ'' to advance E_j.
            EstimateTuple candidate = new EstimateTuple(reached.m(), reached.d() + 1);
            if (candidate.compareTo(best) < 0) best = candidate;
        }
        return best;
    }

    /**
     * Best {@link EstimateTuple} achievable from {@code fromState} to any
     * marked state in component {@code j}.
     */
    private EstimateTuple bestDistanceToMarked(int j, String fromState) {
        ComponentData cd   = compData.get(j);
        EstimateTuple best = EstimateTuple.UNREACHABLE;

        for (String m : cd.markedStates) {
            Map<String, Integer> distMap = cd.distToEachMarked.get(m);
            if (distMap == null) continue;
            Integer d = distMap.get(fromState);
            if (d == null) continue;  // fromState cannot reach m

            EstimateTuple candidate = new EstimateTuple(mFlag(j, m), d);
            if (candidate.compareTo(best) < 0) best = candidate;
        }
        return best;
    }

    /**
     * Determines the {@code m} value for a component sub-state:
     * {@code -1} if it appears in Goals (RA.G), {@code 0} if already visited,
     * {@code 1} if unvisited.
     */
    private int mFlag(int j, String componentState) {
        if (useG && goalCompStates.get(j).contains(componentState)) return -1;
        if (visitedCompStates.get(j).contains(componentState)) return 0;
        return 1;
    }

    // ── utility ───────────────────────────────────────────────────────────────

    /**
     * Returns true iff the composite state represented by {@code parts}
     * satisfies the global marking condition: every component with a non-empty
     * marking set has its current sub-state inside that set.
     */
    private boolean isMarkedComposite(String[] parts) {
        for (int j = 0; j < numComponents; j++) {
            Set<String> marked = compData.get(j).markedStates;
            if (!marked.isEmpty()
                    && (j >= parts.length || !marked.contains(parts[j]))) {
                return false;
            }
        }
        return true;
    }

    private String[] splitCompositeState(String s) {
        return s.split("\\|", numComponents);
    }

    /** Returns a new list with the elements sorted in descending order. */
    private List<EstimateTuple> sortedDescending(List<EstimateTuple> list) {
        List<EstimateTuple> copy = new ArrayList<>(list);
        copy.sort(Comparator.reverseOrder());
        return copy;
    }

    /** Lexicographic comparison of two equal-or-different-length tuple lists. */
    private int compareLex(List<EstimateTuple> a, List<EstimateTuple> b) {
        int len = Math.min(a.size(), b.size());
        for (int i = 0; i < len; i++) {
            int c = a.get(i).compareTo(b.get(i));
            if (c != 0) return c;
        }
        return Integer.compare(a.size(), b.size());
    }
}
