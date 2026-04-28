package newmtsa.synthesis.heuristics.ra;

import newmtsa.parser.ast.LTS;
import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.heuristics.Heuristic;
import newmtsa.synthesis.heuristics.SynthesisContext;

import java.util.*;
import java.util.Arrays;

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
    private final boolean useR;         // RA.R: recompute estimates on new marked-state discoveries
    private final boolean useE;         // RA.E: structure-aware tie-breaking
    private final boolean useG;         // RA.G: use Goals as additional targets
    private final boolean useOpenQueue; // RA.Open: restrict picks to open states only

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

    private RAHeuristic(boolean useR, boolean useE, boolean useG, boolean useOpenQueue) {
        this.useR          = useR;
        this.useE          = useE;
        this.useG          = useG;
        this.useOpenQueue  = useOpenQueue;
    }

    /** Plain RA — corrected formulation (Pazos 2024). */
    public static RAHeuristic base()    { return new RAHeuristic(false, false, false, false); }
    /** RA.R — recompute estimates when new marked states are discovered. */
    public static RAHeuristic withR()   { return new RAHeuristic(true,  false, false, false); }
    /** RA.E — structure-aware tie-breaking. */
    public static RAHeuristic withE()   { return new RAHeuristic(false, true,  false, false); }
    /** RA.ER — recompute + structure-aware (best single-improvement combination). */
    public static RAHeuristic withER()  { return new RAHeuristic(true,  true,  false, false); }
    /** RA.ERG — all improvements enabled. */
    public static RAHeuristic withERG() { return new RAHeuristic(true,  true,  true,  false); }

    /**
     * Returns a new instance identical to this one but with the open queue enabled.
     *
     * <p>The open queue (from Ciolek's original OTF-DCS, §5.1.3) restricts the
     * heuristic's pick to transitions from <em>open</em> states only.  A state is
     * open when all its remaining pending transitions are controllable; a state
     * with at least one pending uncontrollable transition is <em>closed</em> until
     * its descendants are classified (Goal/Error) or a cycle returns to it —
     * at which point it naturally reopens as those transitions leave {@code pending}.
     *
     * <p>This implements a depth-first bias: after exploring a branch from state A,
     * if A still has uncontrollable transitions, they are deferred until the current
     * branch is resolved.  States with only controllable remaining stay open
     * (competing with their descendants) because the controller can freely choose
     * not to take those branches — no confirmation is needed.
     *
     * <p>Pazos (2024) chose not to include the open queue in the corrected RA
     * (preferring simplicity and the RA.E improvement which subsumes much of its
     * benefit), but evaluated it as the {@code RA.Open} variant in §7.
     */
    public RAHeuristic withOpenQueue() {
        return new RAHeuristic(useR, useE, useG, true);
    }

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
    public int pick(List<ExtendedTransition> pending) {
        if (ctx == null) return 0;  // not yet initialised — safe fallback

        refreshVisitedStates();
        if (useG) refreshGoalStates();

        // Restrict to open states when the open-queue flag is enabled.
        List<ExtendedTransition> candidates = useOpenQueue ? openQueueCandidates(pending) : pending;
        if (candidates.isEmpty()) candidates = pending;  // fallback: all closed, use full pending

        // Rebuild RA graph estimates for the candidate set before comparing.
        applyRAGraphPropagation(candidates);

        // Linear scan: find the transition that is "smallest" under the RA ordering.
        ExtendedTransition best = null;
        for (ExtendedTransition t : candidates) {
            if (best == null || compareTransitions(t, best) < 0) {
                best = t;
            }
        }
        lastPicked = best;
        int idx = pending.indexOf(best);
        return idx >= 0 ? idx : 0;
    }

    // ── open queue ───────────────────────────────────────────────────────────

    /**
     * States that are currently <em>closed</em> (excluded from picks).
     *
     * <p>A state is closed when the heuristic last picked a transition from it
     * and it still had at least one uncontrollable transition remaining in
     * {@code pending}.  It reopens naturally once all its uncontrollable
     * transitions leave {@code pending} (because the synthesis engine classified
     * their source or target states and the pruning step removed them).
     */
    private final Set<String> closedStates = new HashSet<>();

    /** The transition returned by the most recent {@link #pick} call. */
    private ExtendedTransition lastPicked = null;

    /**
     * Maintains the open/closed state sets and returns the open-queue view of
     * {@code pending} (transitions from states not in {@link #closedStates}).
     *
     * <p><b>Closing rule</b> (Ciolek §5.1.3, Pazos §2.3.1): after picking from
     * state A, if A still has at least one uncontrollable transition in
     * {@code pending}, A is closed.  The environment could force those
     * transitions, so the algorithm defers them until the current branch is
     * resolved.  If A has only controllable transitions remaining, it stays open
     * — the controller can freely choose not to take them.
     *
     * <p><b>Reopening rule</b>: a closed state reopens when all its uncontrollable
     * transitions have left {@code pending}.  This happens implicitly whenever the
     * synthesis engine's classification-and-prune step removes those transitions
     * (their source or target became a Goal or Error).
     *
     * @return transitions from open states; falls back to full {@code pending} if
     *         every state is closed (prevents deadlock)
     */
    private List<ExtendedTransition> openQueueCandidates(List<ExtendedTransition> pending) {
        // Step 1: close the source of the last picked transition if it still has
        // uncontrollable transitions in pending.
        if (lastPicked != null) {
            String src = lastPicked.from();
            for (ExtendedTransition t : pending) {
                if (t.from().equals(src) && !controllable.contains(t.action())) {
                    closedStates.add(src);
                    break;
                }
            }
        }

        // Step 2: reopen closed states whose uncontrollable transitions have all
        // left pending (classification pruning removed them).
        closedStates.removeIf(s -> {
            for (ExtendedTransition t : pending) {
                if (t.from().equals(s) && !controllable.contains(t.action())) return false;
            }
            return true;  // no uncontrollable remaining → reopen
        });

        // Step 3: build the open-queue view.
        List<ExtendedTransition> open = new ArrayList<>();
        for (ExtendedTransition t : pending) {
            if (!closedStates.contains(t.from())) open.add(t);
        }
        return open;
    }

    /**
     * Returns the current open-queue size for the given {@code pending} list.
     * Useful for testing and instrumentation; does not modify internal state.
     */
    public int openQueueSize(List<ExtendedTransition> pending) {
        if (!useOpenQueue) return pending.size();
        // Count transitions from states with no pending uncontrollable (closed or not).
        Set<String> closed = new HashSet<>(closedStates);
        return (int) pending.stream()
                .filter(t -> !closed.contains(t.from()))
                .count();
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

        // RA.E: structural tie-breaking applies only to controllable transitions — by design
        // (Pazos 2024 §4.2): for uncontrollable transitions the environment decides, so
        // structure-awareness on the controller side is not meaningful.
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

    // ── estimate computation (RA graph + gap + fixpoint) ─────────────────────

    /** Returns the cached estimate, computing it on first access. */
    private List<EstimateTuple> getOrComputeEstimate(ExtendedTransition t) {
        return estimateCache
                .computeIfAbsent(t.from(), k -> new HashMap<>())
                .computeIfAbsent(t.action(),
                                 k -> computeEstimate(t.from(), t.action()));
    }

    /**
     * Computes the RA estimate for {@code action} from {@code compositeState}
     * using the full RA graph + gap function + fixpoint propagation.
     *
     * Steps:
     *  1. Direct estimate: per-component distance to marked after firing action.
     *  2. RA graph: build enabling edges among all actions currently in the
     *     frontier (from pending, available via the cached estimateCache keys).
     *  3. Fixpoint: propagate via gap function until stable.
     */
    private List<EstimateTuple> computeEstimate(String compositeState, String action) {
        // Direct per-component estimate (same as before, forms Phase 1 baseline).
        return directEstimate(compositeState, action);
    }

    /**
     * Called once per pick() invocation to propagate RA graph estimates across
     * the entire frontier before individual transitions are compared.
     *
     * We rebuild estimates for all (compositeState, action) pairs in {@code pending}:
     *  Phase 1 — direct component estimates (baseline per pair).
     *  Phase 2 — RA graph enabling edges (s_l, l) → (s_t, t) + gap function.
     *  Phase 3 — Bellman-Ford fixpoint: estimate(l) = min(estimate(l), gap(l,t) + estimate(t)).
     *
     * Key format: compositeState + "\0" + action. One entry per (state, action) occurrence,
     * so that the same action at different composite states is handled independently —
     * each occurrence has its own component sub-states and enabling relationships.
     */
    private void applyRAGraphPropagation(List<ExtendedTransition> pending) {
        // Collect unique (compositeState, action) pairs and pre-split their parts.
        Map<String, String[]> keyToParts = new LinkedHashMap<>();
        for (ExtendedTransition t : pending) {
            String key = t.from() + "\0" + t.action();
            keyToParts.putIfAbsent(key, splitCompositeState(t.from()));
        }
        List<String> keys = new ArrayList<>(keyToParts.keySet());

        // Phase 1: direct per-component estimates for every (compositeState, action) pair.
        Map<String, int[]> estPerComp = new LinkedHashMap<>();
        Map<String, int[]> mPerComp   = new LinkedHashMap<>();
        for (String key : keys) {
            String[] parts  = keyToParts.get(key);
            String   action = key.substring(key.indexOf('\0') + 1);
            int[] ds = new int[numComponents];
            int[] ms = new int[numComponents];
            Arrays.fill(ds, Integer.MAX_VALUE / 2);
            Arrays.fill(ms, 2);
            for (int j = 0; j < numComponents; j++) {
                ComponentData cd = compData.get(j);
                if (cd.markedStates.isEmpty()) continue;
                String e_j = (j < parts.length) ? parts[j] : null;
                if (e_j == null) continue;
                EstimateTuple et = directComponentEstimate(j, e_j, action);
                ms[j] = et.m();
                ds[j] = et.d();
            }
            estPerComp.put(key, ds);
            mPerComp.put(key, ms);
        }

        // Phase 2: RA graph — enabling edges (s_l, l) → (s_t, t).
        // Edge exists when, from the specific sub-states of s_l, firing l can eventually enable t.
        Map<String, List<String>> predecessors = new HashMap<>();
        for (String key : keys) predecessors.put(key, new ArrayList<>());

        for (int li = 0; li < keys.size(); li++) {
            String   keyL   = keys.get(li);
            String[] partsL = keyToParts.get(keyL);
            String   actL   = keyL.substring(keyL.indexOf('\0') + 1);
            for (int ti = 0; ti < keys.size(); ti++) {
                if (li == ti) continue;
                String keyT = keys.get(ti);
                String actT = keyT.substring(keyT.indexOf('\0') + 1);
                if (isEnablingEdge(actL, actT, partsL)) {
                    predecessors.get(keyT).add(keyL);
                }
            }
        }

        // Phase 3: Bellman-Ford fixpoint propagation.
        Queue<String> workQueue = new ArrayDeque<>(keys);
        Set<String>   inQueue   = new HashSet<>(keys);
        int maxIter = keys.size() * keys.size() + 1;
        while (!workQueue.isEmpty() && maxIter-- > 0) {
            String keyT = workQueue.poll();
            inQueue.remove(keyT);
            int[]  est_t = estPerComp.get(keyT);
            int[]  m_t   = mPerComp.get(keyT);
            String actT  = keyT.substring(keyT.indexOf('\0') + 1);

            for (String keyL : predecessors.get(keyT)) {
                String[] partsL = keyToParts.get(keyL);
                String   actL   = keyL.substring(keyL.indexOf('\0') + 1);
                int[] est_l = estPerComp.get(keyL);
                int[] m_l   = mPerComp.get(keyL);
                boolean improved = false;

                for (int j = 0; j < numComponents; j++) {
                    ComponentData cd = compData.get(j);
                    if (cd.markedStates.isEmpty()) continue;
                    String e_j = (j < partsL.length) ? partsL[j] : null;
                    if (e_j == null) continue;

                    int gap = gapComponent(j, e_j, actL, actT);
                    if (gap == Integer.MAX_VALUE / 2) continue;

                    int candidate_d = saturatingAdd(gap, est_t[j]);
                    int candidate_m = m_t[j];
                    if (candidate_m < m_l[j]
                            || (candidate_m == m_l[j] && candidate_d < est_l[j])) {
                        est_l[j] = candidate_d;
                        m_l[j]   = candidate_m;
                        improved  = true;
                    }
                }
                if (improved && !inQueue.contains(keyL)) {
                    workQueue.add(keyL); inQueue.add(keyL);
                }
            }
        }

        // Write back to estimate cache.
        for (String key : keys) {
            int    sep    = key.indexOf('\0');
            String cs     = key.substring(0, sep);
            String action = key.substring(sep + 1);
            int[]  ds     = estPerComp.get(key);
            int[]  ms     = mPerComp.get(key);
            List<EstimateTuple> tuples = new ArrayList<>();
            for (int j = 0; j < numComponents; j++) {
                if (compData.get(j).markedStates.isEmpty()) continue;
                tuples.add(ds[j] >= Integer.MAX_VALUE / 2
                        ? EstimateTuple.UNREACHABLE
                        : new EstimateTuple(ms[j], ds[j]));
            }
            estimateCache.computeIfAbsent(cs, k -> new HashMap<>()).put(action, tuples);
        }
    }

    /**
     * Returns true iff action {@code l} is an enabling predecessor of {@code t}
     * in some component: specifically, l's successor in component j can reach a
     * state where t is enabled, but t is not currently enabled at e_j.
     */
    private boolean isEnablingEdge(String l, String t, String[] parts_l) {
        for (int j = 0; j < numComponents; j++) {
            ComponentData cd = compData.get(j);
            String e_j = (j < parts_l.length) ? parts_l[j] : null;
            if (e_j == null) continue;
            // t must NOT be enabled at e_j already.
            if (cd.enabledAt.getOrDefault(e_j, Set.of()).contains(t)) continue;
            // l must fire in component j (non-self-loop).
            String succ_l = cd.trans.getOrDefault(l, Map.of()).get(e_j);
            if (succ_l == null || succ_l.equals(e_j)) continue;
            // From succ_l, can t eventually become enabled?
            Set<String> reachable = cd.reachableFromState.getOrDefault(succ_l, Set.of());
            for (String rs : reachable) {
                if (cd.enabledAt.getOrDefault(rs, Set.of()).contains(t)) return true;
            }
        }
        return false;
    }

    /**
     * Gap cost for component {@code j} when firing {@code l} to eventually enable {@code t}.
     *
     * = stepsToT (steps from l's successor to the nearest t-enabling state)
     *   + dist(t-enabling state → nearest marked state in j).
     *
     * We iterate over all t-enabling states reachable from l's successor and take the
     * minimum dist-to-marked, then add stepsToT as the shared approach cost.
     *
     * Returns {@code Integer.MAX_VALUE / 2} if t cannot be enabled from l's successor.
     */
    private int gapComponent(int j, String e_j, String l, String t) {
        ComponentData cd = compData.get(j);
        String succ_l = cd.trans.getOrDefault(l, Map.of()).get(e_j);
        if (succ_l == null || succ_l.equals(e_j)) return Integer.MAX_VALUE / 2;

        Integer stepsToT = cd.stepsToEnableAction.getOrDefault(t, Map.of()).get(succ_l);
        if (stepsToT == null) return Integer.MAX_VALUE / 2;

        // Iterate over t-enabling states reachable from succ_l; use the one with the
        // best (minimum) distance to marked. gap = stepsToT + dist(t-enabling state → marked).
        Set<String> reachable = cd.reachableFromState.getOrDefault(succ_l, Set.of());
        int bestDist = Integer.MAX_VALUE / 2;
        for (Map.Entry<String, Set<String>> entry : cd.enabledAt.entrySet()) {
            String s = entry.getKey();
            if (!entry.getValue().contains(t)) continue;
            if (!reachable.contains(s)) continue;
            EstimateTuple dt = bestDistanceToMarked(j, s);
            if (dt != EstimateTuple.UNREACHABLE && dt.d() < bestDist) bestDist = dt.d();
        }
        if (bestDist == Integer.MAX_VALUE / 2) return Integer.MAX_VALUE / 2;
        return saturatingAdd(stepsToT, bestDist);
    }

    private static int saturatingAdd(int a, int b) {
        long r = (long) a + b;
        return r >= Integer.MAX_VALUE / 2 ? Integer.MAX_VALUE / 2 : (int) r;
    }

    /** Direct per-component estimate without RA graph propagation. */
    private List<EstimateTuple> directEstimate(String compositeState, String action) {
        String[]            parts  = splitCompositeState(compositeState);
        List<EstimateTuple> result = new ArrayList<>();
        for (int j = 0; j < numComponents; j++) {
            if (compData.get(j).markedStates.isEmpty()) continue;
            String e_j = (j < parts.length) ? parts[j] : null;
            result.add(e_j == null ? EstimateTuple.UNREACHABLE
                                   : directComponentEstimate(j, e_j, action));
        }
        return result;
    }

    /**
     * Per-component estimate for {@code action} from sub-state {@code e_j}
     * (Definition 8 of Pazos 2024 — direct BFS only, no RA graph).
     */
    private EstimateTuple directComponentEstimate(int j, String e_j, String action) {
        ComponentData cd = compData.get(j);

        String  succ               = cd.trans.getOrDefault(action, Map.of()).get(e_j);
        boolean isSelfloopOrAbsent = (succ == null || succ.equals(e_j));

        if (!isSelfloopOrAbsent) {
            // ERA edge weight = |trace starting with action from e_j reaching marked|
            // = 1 (for action itself) + BFS dist from succ to marked.
            EstimateTuple base = bestDistanceToMarked(j, succ);
            return base == EstimateTuple.UNREACHABLE
                   ? EstimateTuple.UNREACHABLE
                   : new EstimateTuple(base.m(), base.d() + 1);
        }

        // Cases 3/4 (Def. 8 Pazos 2024): action absent/self-loop in j, already at marked.
        // d=1 because one composite step (firing action) keeps us at the marked state.
        if (cd.markedStates.contains(e_j)) {
            return new EstimateTuple(mFlag(j, e_j), 1);
        }

        // Cases 5/6: action absent/self-loop, use best alternative event β enabled at e_j.
        // Formula: 1 (for firing action) + d(β, m) in ERA.
        // d(β, m) in ERA = 1 (for β) + BFS dist from succ_β to m  →  total = 2 + BFS dist.
        EstimateTuple best = EstimateTuple.UNREACHABLE;
        for (String alt : cd.enabledAt.getOrDefault(e_j, Set.of())) {
            if (alt.equals(action)) continue;
            String alt_succ = cd.trans.getOrDefault(alt, Map.of()).get(e_j);
            if (alt_succ == null || alt_succ.equals(e_j)) continue;
            EstimateTuple reached = bestDistanceToMarked(j, alt_succ);
            if (reached == EstimateTuple.UNREACHABLE) continue;
            EstimateTuple candidate = new EstimateTuple(reached.m(), reached.d() + 2);
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

    // ── test / inspection API ────────────────────────────────────────────────

    /**
     * Returns the RA estimate for the given {@code (compositeState, action)} pair,
     * including full RA graph propagation across the single-transition frontier.
     * Intended for unit tests and diagnostic inspection.
     *
     * <p>Requires {@link #init(SynthesisContext)} to have been called first.
     */
    public List<EstimateTuple> estimateFor(String compositeState, String action) {
        refreshVisitedStates();
        if (useG) refreshGoalStates();
        ExtendedTransition t = new ExtendedTransition(compositeState, action, "?");
        applyRAGraphPropagation(List.of(t));
        return estimateCache.getOrDefault(compositeState, Map.of())
                            .getOrDefault(action, List.of());
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
        return ctx.plantStateKey(s).split("\\|", numComponents);
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
        // TODO: the correct tie-breaking when lists have different lengths is unclear —
        // Pazos 2024 does not specify this case. Currently shorter = better; revisit once clarified.
        return Integer.compare(a.size(), b.size());
    }
}
