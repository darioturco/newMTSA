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
    /** RA.ERG — all improvements enabled (R + E + G + open queue tie-break). */
    public static RAHeuristic withERG() { return new RAHeuristic(true,  true,  true,  true); }

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

        currentPending = pending;
        refreshVisitedStates();
        if (useG) refreshGoalStates();

        // Rebuild RA graph estimates over full pending.
        applyRAGraphPropagation(pending);

        // Sync open queue state (track classifications, discoveries) without hard filtering.
        if (useOpenQueue) openQueueCandidates(pending);

        if (ctx.verbose() && useOpenQueue) {
            System.out.println("[OQ] pending=" + pending.size()
                    + " openStates=" + openStates.size() + " closedStates=" + closedStates.size()
                    + " classified=" + (ctx.goals().size() + ctx.errors().size()));
        }

        // Pick best transition. With useOpenQueue, restrict candidates to transitions
        // from open states (hard filter, mirroring original MTSA OpenSetExplorationHeuristic
        // where closed compostates are absent from the priority queue). Sticky:
        // if no open candidates exist (deadlock), fall back to full pending to keep
        // exploration alive.
        List<ExtendedTransition> candidates = pending;
        if (useOpenQueue) {
            List<ExtendedTransition> open = new ArrayList<>(pending.size());
            for (ExtendedTransition t : pending) {
                if (openStates.contains(t.from())) open.add(t);
            }
            if (!open.isEmpty()) candidates = open;
        }

        ExtendedTransition best = null;
        for (ExtendedTransition t : candidates) {
            if (best == null) { best = t; continue; }
            if (compareTransitions(t, best) < 0) best = t;
        }
        lastPicked = best;

        // After pick: if state has unresolved uncontrollables remaining, close it (defer).
        if (useOpenQueue && best != null) {
            updateOpenStatePostPick(best, pending);
        }

        if (best == null) return 0;
        // Map back to pending index (candidates may be a filtered sub-list).
        int idx = pending.indexOf(best);
        return idx >= 0 ? idx : 0;
    }

    // ── state-level open queue ───────────────────────────────────────────────

    /**
     * Composite states currently open for expansion.
     * A state is open when picking from it does not violate the DFS bias
     * (i.e., its already-fired uncontrollable children are all classified).
     */
    private final Set<String> openStates = new LinkedHashSet<>();

    /** States deferred (have unresolved uncontrollable transitions). */
    private final Set<String> closedStates = new HashSet<>();

    /** All states ever seen in any pending list. */
    private final Set<String> knownStates = new HashSet<>();

    /** The transition returned by the most recent {@link #pick} call. */
    private ExtendedTransition lastPicked = null;

    /** Snapshot of the pending list for the current {@link #pick} invocation. */
    private List<ExtendedTransition> currentPending = List.of();

    /**
     * Compute the open-queue view of {@code pending}: transitions from open states only.
     *
     * <p>State-level open queue (mirrors original MTSA OpenSetExplorationHeuristic):
     * <ul>
     *   <li>Newly-discovered states from pending → added to {@link #openStates}.</li>
     *   <li>Classified states (in goals/errors) → removed; parents conditionally reopened.</li>
     *   <li>States already in {@link #closedStates} → excluded from candidates.</li>
     * </ul>
     */
    private List<ExtendedTransition> openQueueCandidates(List<ExtendedTransition> pending) {
        // Step 1: discover new states from pending.
        for (ExtendedTransition t : pending) {
            String src = t.from();
            if (knownStates.add(src)) {
                if (!isClassified(src)) openStates.add(src);
            }
        }

        // Step 2: process newly classified states (recursive reopen of NONE parents).
        for (String s : ctx.goals())  processClassified(s);
        for (String s : ctx.errors()) processClassified(s);

        // Step 3: build candidates from open states (excluding closed and classified).
        List<ExtendedTransition> open = new ArrayList<>();
        for (ExtendedTransition t : pending) {
            if (openStates.contains(t.from())) open.add(t);
        }
        return open;
    }

    /** True if state is in goals or errors set. */
    private boolean isClassified(String s) {
        return ctx.goals().contains(s) || ctx.errors().contains(s);
    }

    /**
     * Process state {@code s} that has been classified.
     * Removes {@code s} from open/closed, then re-opens its NONE predecessors
     * via the recursive {@link #reopen} logic (mirrors original MTSA
     * {@code OpenSetExplorationHeuristic.open}).
     */
    private void processClassified(String s) {
        openStates.remove(s);
        closedStates.remove(s);
        for (String p : ctx.predecessorsOf(s)) {
            if (isClassified(p)) continue;
            reopen(p);
        }
    }

    /**
     * Reopens {@code s} according to the original MTSA {@code open()} recursion:
     * <ol>
     *   <li>If {@code s} is already open or classified, do nothing.</li>
     *   <li>If {@code s} has no still-NONE explored child with pending transitions,
     *       add {@code s} to {@link #openStates}.</li>
     *   <li>Otherwise, recursively try to reopen each such NONE child first
     *       (DFS bias — prefer deeper exploration before re-firing {@code s}'s
     *       own remaining actions). Only add {@code s} to {@link #openStates}
     *       if either no child was opened OR {@code s} is a controllable state
     *       ({@code state.isControlled()} in the original).</li>
     * </ol>
     *
     * @return true iff {@code s} (or one of its NONE descendants) was opened.
     */
    private boolean reopen(String s) {
        if (isClassified(s)) return false;
        if (openStates.contains(s)) return true;

        List<String> noneChildren = new ArrayList<>();
        for (ExtendedTransition t : ctx.successorsOf(s)) {
            String c = t.to();
            if (c.equals(s)) continue;
            if (!ctx.exploredStates().contains(c)) continue;  // not yet explored
            if (isClassified(c)) continue;
            if (openStates.contains(c)) continue;
            if (!hasPendingFromState(c)) continue;             // fullyExplored
            if (!noneChildren.contains(c)) noneChildren.add(c);
        }

        boolean childOpened = false;
        for (String c : noneChildren) {
            if (reopen(c)) childOpened = true;
        }

        if (!childOpened || isControllableState(s)) {
            closedStates.remove(s);
            openStates.add(s);
            return true;
        }
        return childOpened;
    }

    /** True iff at least one transition in the current pending frontier has {@code s} as source. */
    private boolean hasPendingFromState(String s) {
        for (ExtendedTransition t : currentPending) {
            if (t.from().equals(s)) return true;
        }
        return false;
    }

    /**
     * Called after a transition is picked. Defers the source state if it still
     * has any uncontrollable in pending (DFS bias — defer until current branch resolved).
     * Mirrors original MTSA OpenSetExplorationHeuristic.expansionDone():
     * "if state is controlled and still NONE, re-queue it" — i.e., close otherwise.
     */
    private void updateOpenStatePostPick(ExtendedTransition picked, List<ExtendedTransition> pending) {
        String src = picked.from();
        if (isClassified(src)) {
            openStates.remove(src);
            return;
        }
        // Mirror old MTSA Compostate.isControlled(): state is "controlled" iff it has
        // NO uncontrollable transition in its alphabet at all. Old expansionDone() only
        // re-adds a state to the open queue when state.isControlled(); otherwise the
        // state stays closed (deferred) until a child gets classified. This is the
        // DFS bias.
        if (!isControllableState(src)) {
            openStates.remove(src);
            closedStates.add(src);
        }
        // Else: pure controllable state — stays open.
    }

    /** Inspection API for tests. Returns count of transitions from open states. */
    public int openQueueSize(List<ExtendedTransition> pending) {
        if (!useOpenQueue) return pending.size();
        return (int) pending.stream()
                .filter(t -> openStates.contains(t.from()) || !knownStates.contains(t.from()))
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
        int cmp = aCtrl ? lex : -lex;
        if (cmp != 0) return cmp;

        // Depth tiebreaker: shallower first for both (matches original CompostateRanker c1.depth - c2.depth).
        int depthA = ctx.depthOf(a.from());
        int depthB = ctx.depthOf(b.from());
        if (depthA >= 0 && depthB >= 0 && depthA != depthB) {
            return depthA - depthB;
        }
        return 0;
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
     * Returns true iff no controllable transition from {@code from} has been
     * expanded (removed from pending and processed) — mirrors
     * {@code Compostate.getControllablesExpandedCount() == 0}.
     *
     * A transition is "expanded" when its target has been explored AND the
     * transition is no longer in the current pending list (it was picked earlier).
     */
    private boolean noExploredControllableChildren(String from) {
        for (ExtendedTransition t : ctx.successorsOf(from)) {
            if (!controllable.contains(t.action())) continue;
            if (!ctx.exploredStates().contains(t.to())) continue;
            boolean stillPending = false;
            for (ExtendedTransition p : currentPending) {
                if (p.from().equals(from) && p.action().equals(t.action())) {
                    stillPending = true;
                    break;
                }
            }
            if (!stillPending) return false;
        }
        return true;
    }

    /**
     * A composite state is "controllable" if every known successor transition is
     * controllable (the environment has no forced moves from this state).
     *
     * <p>Uses the full successor list rather than pending, because a pending-only
     * check over-promotes states whose uncontrollable successors were classified
     * early (via backpropagation), leading to excessive structural priority when
     * combined with per-component marking estimates.
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
     * <p>Evaluates RA per source composite state (mirroring original MTSA
     * {@code ReadyAbstraction.eval(compostate)}): for each pending source
     * {@code cs} not yet in the cache, build a fresh RA graph over its
     * <em>locally-enabled-non-self-loop</em> vertex set (across all components),
     * run Bellman-Ford propagation with gap/readyInLTS guard, apply
     * reconcilateShort, and cache the estimate for every globally-enabled action.
     *
     * <p>Crucially the vertex set includes actions that are NOT in the current
     * pending frontier (e.g. {@code accept}, {@code reject}, intermediate
     * {@code put}/{@code return}) — these "phantom" vertices carry the
     * propagation chain from marked-reaching actions back through to the
     * controllable frontier actions, just as in the original MTSA.
     */
    private void applyRAGraphPropagation(List<ExtendedTransition> pending) {
        // Group pending transitions by their source composite state, preserving order.
        Map<String, List<String>> pendingActionsBySource = new LinkedHashMap<>();
        for (ExtendedTransition t : pending) {
            pendingActionsBySource
                    .computeIfAbsent(t.from(), k -> new ArrayList<>())
                    .add(t.action());
        }

        for (Map.Entry<String, List<String>> entry : pendingActionsBySource.entrySet()) {
            String cs = entry.getKey();
            if (estimateCache.containsKey(cs)) continue;   // already evaluated (isEvaluated guard)
            evaluateCompostate(cs, new LinkedHashSet<>(entry.getValue()));
        }
    }

    /**
     * Evaluates the RA estimate for a single composite state {@code cs}.
     * Mirrors original {@code ReadyAbstraction.eval} + {@code buildRA} +
     * {@code evaluateRA} + {@code reconcilateShort}.
     */
    private void evaluateCompostate(String cs, Set<String> availableActions) {
        String[] parts = splitCompositeState(cs);

        // ── Build vertices: actions that fire non-self-loop in some component at parts[j] ──
        // Also collect readyInLTS[vertex] = {j : vertex non-self-loop in j at parts[j]}.
        Map<String, Set<Integer>> readyInLTS = new LinkedHashMap<>();
        for (int j = 0; j < numComponents; j++) {
            ComponentData cd = compData.get(j);
            String e_j = (j < parts.length) ? parts[j] : null;
            if (e_j == null) continue;
            for (String act : cd.enabledAt.getOrDefault(e_j, Set.of())) {
                String succ = cd.trans.getOrDefault(act, Map.of()).get(e_j);
                if (succ == null || succ.equals(e_j)) continue;   // self-loop or absent
                readyInLTS.computeIfAbsent(act, k -> new HashSet<>()).add(j);
            }
        }
        if (readyInLTS.isEmpty()) {
            // No vertices — cache empty estimates for available actions and return.
            Map<String, List<EstimateTuple>> bucket = new HashMap<>();
            for (String a : availableActions) bucket.put(a, List.of());
            estimateCache.put(cs, bucket);
            return;
        }

        // ── Estimate arrays per vertex (indexed by vertex name) ──
        // Layout: ds[v][j], ms[v][j] over all components (only marked components matter).
        Map<String, int[]> dsByVertex = new HashMap<>();
        Map<String, int[]> msByVertex = new HashMap<>();
        for (String v : readyInLTS.keySet()) {
            int[] ds = new int[numComponents];
            int[] ms = new int[numComponents];
            Arrays.fill(ds, Integer.MAX_VALUE / 2);
            Arrays.fill(ms, 2);
            dsByVertex.put(v, ds);
            msByVertex.put(v, ms);
        }

        // shortest[j] tracks min HDist over registered (vertex,j) where vertex is available.
        int[] shortestD = new int[numComponents];
        int[] shortestM = new int[numComponents];
        Arrays.fill(shortestD, Integer.MAX_VALUE / 2);
        Arrays.fill(shortestM, 2);

        Deque<String> fresh = new ArrayDeque<>();
        Set<String>   inFresh = new HashSet<>();

        // ── Phase 1: per-LTS first loop ──
        // For each marked LTS j, iterate local transitions at parts[j]. For each
        // (l, target) where target is marked: set direct est. For others: search
        // markedReachable from target. Mirrors original first loop in evaluateRA.
        for (int j = 0; j < numComponents; j++) {
            ComponentData cd = compData.get(j);
            if (cd.markedStates.isEmpty()) continue;
            String e_j = (j < parts.length) ? parts[j] : null;
            if (e_j == null) continue;

            for (String l : cd.enabledAt.getOrDefault(e_j, Set.of())) {
                String target = cd.trans.getOrDefault(l, Map.of()).get(e_j);
                if (target == null) continue;
                if (l.equals(target)) continue;   // sentinel safety (shouldn't happen)

                int mt, dt;
                if (cd.markedStates.contains(target)) {
                    mt = mFlag(j, target);
                    dt = 1;
                } else if (e_j.equals(target)) {
                    continue;   // self-loop with non-marked target: skip
                } else {
                    EstimateTuple base = bestDistanceToMarked(j, target);
                    if (base == EstimateTuple.UNREACHABLE) continue;
                    mt = base.m();
                    dt = base.d() + 1;
                }

                int[] ds = dsByVertex.get(l);
                int[] ms = msByVertex.get(l);
                if (ds == null) continue;          // l not in vertex set (self-loop everywhere)
                if (mt < ms[j] || (mt == ms[j] && dt < ds[j])) {
                    ms[j] = mt;
                    ds[j] = dt;
                    if (!inFresh.contains(l)) { fresh.add(l); inFresh.add(l); }
                    if (availableActions.contains(l)) {
                        if (mt < shortestM[j] || (mt == shortestM[j] && dt < shortestD[j])) {
                            shortestM[j] = mt;
                            shortestD[j] = dt;
                        }
                    }
                }
            }
        }

        // ── Phase 2: Build edges within vertices ──
        // edge l → t iff l is in vertices AND in some LTS j (containing both l and t),
        // from succ_l(j), t can eventually become enabled. Mirrors original buildRA.
        Map<String, List<String>> predOfT = new HashMap<>();
        for (String t : readyInLTS.keySet()) predOfT.put(t, new ArrayList<>());
        List<String> vertices = new ArrayList<>(readyInLTS.keySet());
        for (String l : vertices) {
            for (String t : vertices) {
                if (l.equals(t)) continue;
                if (isEnablingEdge(l, t, parts)) {
                    predOfT.get(t).add(l);
                }
            }
        }

        // ── Phase 3: Bellman-Ford propagation from fresh ──
        int maxIter = vertices.size() * vertices.size() + 1;
        while (!fresh.isEmpty() && maxIter-- > 0) {
            String t = fresh.poll();
            inFresh.remove(t);
            int[] est_t = dsByVertex.get(t);
            int[] m_t   = msByVertex.get(t);

            for (String l : predOfT.getOrDefault(t, List.of())) {
                int[]        est_l   = dsByVertex.get(l);
                int[]        m_l     = msByVertex.get(l);
                Set<Integer> lReady  = readyInLTS.get(l);

                int gap = computeGlobalGap(parts, l, t);
                boolean improved = false;

                for (int j = 0; j < numComponents; j++) {
                    if (compData.get(j).markedStates.isEmpty()) continue;
                    if (lReady.contains(j)) continue;        // direct estimate authoritative for l in j
                    if (est_t[j] >= Integer.MAX_VALUE / 2) continue;

                    int candidate_d = saturatingAdd(gap, est_t[j]);
                    int candidate_m = m_t[j];
                    if (candidate_m < m_l[j]
                            || (candidate_m == m_l[j] && candidate_d < est_l[j])) {
                        est_l[j] = candidate_d;
                        m_l[j]   = candidate_m;
                        improved = true;
                        if (availableActions.contains(l)) {
                            if (candidate_m < shortestM[j]
                                    || (candidate_m == shortestM[j] && candidate_d < shortestD[j])) {
                                shortestM[j] = candidate_m;
                                shortestD[j] = candidate_d;
                            }
                        }
                    }
                }
                if (improved && !inFresh.contains(l)) {
                    fresh.add(l);
                    inFresh.add(l);
                }
            }
        }

        // ── Phase 4: reconcilateShort — fill missing shortest then chasm slots ──
        // First, fill shortest[j] for components where no available action registered.
        // Mirrors original reconcilateShort loop 1 (uses markedReachableStates fallback).
        for (int j = 0; j < numComponents; j++) {
            ComponentData cd = compData.get(j);
            if (cd.markedStates.isEmpty()) continue;
            if (shortestM[j] < 2) continue;            // already set
            String e_j = (j < parts.length) ? parts[j] : null;
            if (e_j == null) continue;

            // BFS over labels reachable from e_j to any marked state.
            // shortest = best (m, d) over (markedState m_dest, label l) where d=dist(e_j, m_dest)
            // restricted to labels currently enabled — old uses entry value which is map (action, dist).
            int bestM = 2, bestD = Integer.MAX_VALUE / 2;
            for (String m : cd.markedStates) {
                Map<String, Integer> distMap = cd.distToEachMarked.get(m);
                if (distMap == null) continue;
                Integer d = distMap.get(e_j);
                if (d == null) continue;
                int mFlag_m = mFlag(j, m);
                if (mFlag_m < bestM || (mFlag_m == bestM && d < bestD)) {
                    bestM = mFlag_m;
                    bestD = d;
                }
            }
            if (bestM < 2) {
                shortestM[j] = bestM;
                shortestD[j] = bestD;
            }
        }

        // Second, fill chasm slots in each available action's estimate via shortest+1.
        // Mirrors original reconcilateShort loop 2: for each available action l,
        // for each LTS j where l is NOT ready (or is self-loop in j), fill with shortest[j].inc().
        Map<String, List<EstimateTuple>> bucket = new HashMap<>();
        for (String a : availableActions) {
            int[] ds = dsByVertex.get(a);
            int[] ms = msByVertex.get(a);
            List<EstimateTuple> tuples = new ArrayList<>();
            for (int j = 0; j < numComponents; j++) {
                if (compData.get(j).markedStates.isEmpty()) continue;
                int m, d;
                if (ds != null && ms[j] < 2) {
                    m = ms[j];
                    d = ds[j];
                } else if (shortestM[j] < 2) {
                    // Self-loop or not in vertex set for j: fall back to shortest+1.
                    m = shortestM[j];
                    d = saturatingAdd(shortestD[j], 1);
                } else {
                    tuples.add(EstimateTuple.UNREACHABLE);
                    continue;
                }
                tuples.add(new EstimateTuple(m, d));
            }
            bucket.put(a, tuples);
        }
        estimateCache.put(cs, bucket);
    }

    /**
     * Returns true iff action {@code l} is an enabling predecessor of {@code t}
     * in some component: l fires (non-self-loop) in component j and from l's
     * successor, t can eventually become enabled.
     */
    private boolean isEnablingEdge(String l, String t, String[] parts_l) {
        for (int j = 0; j < numComponents; j++) {
            ComponentData cd = compData.get(j);
            String e_j = (j < parts_l.length) ? parts_l[j] : null;
            if (e_j == null) continue;
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
     * Global gap for the RA graph edge (l → t): maximum over all components j
     * where l fires (non-self-loop) AND t is in the alphabet of j, of the
     * steps from l's successor in j to FIRE t (enable it + 1 step to fire it).
     *
     * Matches original MTSA: gap = manyStepsReachableActions[s][t][l] - 1
     *                             = stepsToEnableAction[t][succ_l] + 1.
     * The +1 accounts for the firing step of t itself.
     */
    private int computeGlobalGap(String[] partsL, String actL, String actT) {
        int result = 0;
        for (int j = 0; j < numComponents; j++) {
            ComponentData cd = compData.get(j);
            if (!cd.alphabet.contains(actT)) continue;
            String e_j = (j < partsL.length) ? partsL[j] : null;
            if (e_j == null) continue;
            String succ_l = cd.trans.getOrDefault(actL, Map.of()).get(e_j);
            if (succ_l == null || succ_l.equals(e_j)) continue;
            Integer stepsToT = cd.stepsToEnableAction.getOrDefault(actT, Map.of()).get(succ_l);
            if (stepsToT == null) continue;
            int gapJ = stepsToT + 1;  // +1: cost of firing t itself (matches original gap formula)
            if (gapJ > result) result = gapJ;
        }
        return result;
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

        // Non-marked self-loop: return UNREACHABLE — filled by reconcilateShort
        // in applyRAGraphPropagation using the global shortest estimate + 1.
        return EstimateTuple.UNREACHABLE;
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

    // ── verbose output ───────────────────────────────────────────────────────

    @Override
    public void printFrontier(List<ExtendedTransition> pending, int pickedIndex) {
        if (ctx == null || !ctx.verbose()) return;

        System.out.println("\n=== RA Frontier (size=" + pending.size() + ") ===");
        if (useOpenQueue) {
            System.out.println("Open states: " + openStates.size()
                    + " | Closed states: " + closedStates.size()
                    + " | Classified: " + (ctx.goals().size() + ctx.errors().size()));
        }
        System.out.printf("%-3s %-3s %-25s %-8s %-25s | Estimate%n", "", "Idx", "From", "Action", "To");
        System.out.println("----" + "-".repeat(95));

        for (int i = 0; i < pending.size(); i++) {
            ExtendedTransition t = pending.get(i);
            String mark = (i == pickedIndex) ? " >>> " : "     ";
            String state = useOpenQueue
                    ? (openStates.contains(t.from()) ? "O" : (closedStates.contains(t.from()) ? "C" : "?"))
                    : " ";
            List<EstimateTuple> est = getOrComputeEstimate(t);
            String estStr = est.isEmpty() ? "N/A" : est.toString();
            System.out.printf("%s[%s] %-2d %-25s %-8s %-25s | %s%n",
                    mark, state, i,
                    truncate(t.from(), 25),
                    truncate(t.action(), 8),
                    truncate(t.to(), 25),
                    estStr);
        }
        System.out.println();
    }

    private static String truncate(String s, int len) {
        return s.length() <= len ? s : s.substring(0, len - 2) + "..";
    }

    // ── test / inspection API ────────────────────────────────────────────────

    /**
     * Returns the RA estimate for the given {@code (compositeState, action)} pair,
     * including full RA graph propagation across the single-transition frontier.
     * Intended for unit tests and diagnostic inspection.
     *
     * <p>Requires {@link #init(SynthesisContext)} to have been called first.
     */
    /** Read-only cache accessor — does NOT trigger computation. */
    public List<EstimateTuple> estimateCacheFor(String compositeState, String action) {
        return estimateCache.getOrDefault(compositeState, Map.of())
                            .getOrDefault(action, List.of());
    }

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
