package newmtsa.synthesis.gr1;

import newmtsa.parser.ast.LTS;
import newmtsa.parser.ast.LtlPropertyDef;
import newmtsa.parser.ast.Transition;
import newmtsa.synthesis.Director;
import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.OTFDirectedControlledSynthesis;

import newmtsa.synthesis.features.FeatureCompute;
import newmtsa.synthesis.features.FeaturesContext;
import newmtsa.synthesis.heuristics.Heuristic;
import newmtsa.synthesis.heuristics.SimpleSynthesisContext;

import java.util.*;

/**
 * On-the-fly Directed Controller Synthesis for GR(1) liveness objectives.
 *
 * <p>Winning condition: ∧_i ([]<>G_i) — the controller must guarantee that
 * every liveness property G_0, G_1, ..., G_{k-1} is visited infinitely often.
 *
 * <p><b>Reduction to single Büchi</b>: the k-guarantee game is converted to a
 * single Büchi game via a round-robin phase counter. The extended state is
 * {@code (plant_state, phase)} with {@code phase ∈ {0, …, k-1}}.  State
 * {@code (s, i)} is Büchi-accepting iff G_i holds at {@code s}.  When the
 * controller departs a marked state the phase advances: {@code (i+1) mod k}.
 *
 * <p>The DCS exploration/propagation algorithm is identical to the non-blocking
 * variant; only the state space and the marking predicate differ.
 *
 * <p><b>State encoding</b>: {@code "c0|c1|…|cN|phase"} — component states
 * joined by {@code "|"} followed by the integer phase.  Compound-assert state
 * names use {@code "_"} as their internal separator, so there is no ambiguity
 * when splitting on {@code "|"}.
 */
public class OTFDirectedControledSyntesisGR1 implements OTFDirectedControlledSynthesis {

    // ── inputs ────────────────────────────────────────────────────────────────

    private final List<LTS>   components;
    private final Set<String> controllable;
    public  final Heuristic   heuristic;
    private final boolean     verbose;
    private final boolean     useNumericIds;
    private final boolean     frontierRestriction;

    // ── GR(1) liveness data ───────────────────────────────────────────────────
    //
    // The GR(1) game is reduced to a single Büchi game via a round-robin phase
    // counter j ∈ {0, …, numPhases-1}.  The phase is an algorithmic construct,
    // NOT an LTS component in the parallel composition.
    //
    // Node identity in this algorithm: "c0|c1|…|cN#j"
    //   • "c0|c1|…|cN" is the plant state (parallel composition of components)
    //   • "#j" is the phase suffix, separated by '#' to distinguish it from the
    //     '|'-separated component states.
    //
    // Previously the phase was appended as "|j", making it look like an extra
    // component in the composition.  The '#' separator makes the boundary explicit.
    //
    // Use plantPart(nodeId) to get the plant state and phaseOf(nodeId) to get j.

    private final int   numGuarantees;
    private final int[] guaranteeIndices;  // index into components for each G_i
    private final int   numAssumptions;
    private final int[] assumptionIndices; // index into components for each A_i
    private final int   numPhases;         // numAssumptions + numGuarantees
    private final int[] phaseComponentIndex; // phaseComponentIndex[j] = component index for phase j

    // ── per-component data ────────────────────────────────────────────────────

    private final List<Map<String, Map<String, String>>> compTrans;
    private final List<Set<String>>                      compAlpha;
    private final Set<String>                            alphabet;
    private final List<Set<String>>                      compMarked;

    // ── state classification ──────────────────────────────────────────────────

    private final Set<String> goals  = new LinkedHashSet<>();
    private final Set<String> errors = new LinkedHashSet<>();
    private final Set<String> none   = new LinkedHashSet<>();

    // ── exploration structure ─────────────────────────────────────────────────

    private final Map<String, List<ExtendedTransition>> succMap = new HashMap<>();
    private final Map<String, Set<String>>              parents = new HashMap<>();
    private final Map<String, Integer>                  phaseCache = new HashMap<>();
    private final Map<String, String[]>                 splitStateCache = new HashMap<>();
    private final Map<String, Boolean>                  assumptionSatCache = new HashMap<>();
    private       boolean                               pendingDirty = false;

    // ── feature computation ───────────────────────────────────────────────────

    private final FeatureCompute  featureCompute;
    private final FeaturesContext featuresCtx;

    // ── main-loop state ───────────────────────────────────────────────────────

    private final String                   s0;
    private       List<ExtendedTransition> pending;
    private       boolean                  explorationEnded;
    private       int                      transitionsExplored = 0;

    // ── constructor ───────────────────────────────────────────────────────────

    /**
     * @param components   plant processes + guarantee fluents + assumption fluents
     *                     (in that order, as assembled by the test/caller)
     * @param assumptions  environment liveness assumptions (reserved; only guarantee
     *                     liveness is enforced in this implementation)
     * @param guarantees   liveness guarantees G_0 .. G_{k-1}; each LTS must appear
     *                     in {@code components}
     * @param controllable set of controllable event labels
     * @param heuristic    frontier selection strategy
     * @param verbose      print trace to stdout when {@code true}
     */
    public OTFDirectedControledSyntesisGR1(List<LTS>            components,
                                            List<LtlPropertyDef> assumptions,
                                            List<LtlPropertyDef> guarantees,
                                            Set<String>          controllable,
                                            Heuristic            heuristic,
                                            boolean              verbose) {
        this(components, assumptions, guarantees, controllable, heuristic, verbose, false, null);
    }

    public OTFDirectedControledSyntesisGR1(List<LTS>            components,
                                            List<LtlPropertyDef> assumptions,
                                            List<LtlPropertyDef> guarantees,
                                            Set<String>          controllable,
                                            Heuristic            heuristic,
                                            boolean              verbose,
                                            boolean              useNumericIds) {
        this(components, assumptions, guarantees, controllable, heuristic, verbose, useNumericIds, null);
    }

    public OTFDirectedControledSyntesisGR1(List<LTS>            components,
                                            List<LtlPropertyDef> assumptions,
                                            List<LtlPropertyDef> guarantees,
                                            Set<String>          controllable,
                                            Heuristic            heuristic,
                                            boolean              verbose,
                                            boolean              useNumericIds,
                                            FeatureCompute       featureCompute) {
        this(components, assumptions, guarantees, controllable, heuristic, verbose, useNumericIds, featureCompute, false);
    }

    public OTFDirectedControledSyntesisGR1(List<LTS>            components,
                                            List<LtlPropertyDef> assumptions,
                                            List<LtlPropertyDef> guarantees,
                                            Set<String>          controllable,
                                            Heuristic            heuristic,
                                            boolean              verbose,
                                            boolean              useNumericIds,
                                            FeatureCompute       featureCompute,
                                            boolean              frontierRestriction) {
        if (controllable.isEmpty())
            throw new IllegalArgumentException("DCS requires at least one controllable action");

        this.controllable        = Set.copyOf(controllable);
        this.heuristic           = heuristic;
        this.verbose             = verbose;
        this.useNumericIds       = useNumericIds;
        this.frontierRestriction = frontierRestriction;
        this.numGuarantees  = guarantees.size();
        this.numAssumptions = assumptions.size();
        this.numPhases      = numAssumptions + numGuarantees;
        this.guaranteeIndices    = new int[numGuarantees];
        this.assumptionIndices   = new int[numAssumptions];
        this.phaseComponentIndex = new int[numPhases];

        // Guarantee/assumption LTS created from compound asserts have isFluent=false
        // (parser limitation). Force isFluent=true so they self-loop on unknown actions
        // instead of blocking the parallel composition.
        Set<String> livNames = new LinkedHashSet<>();
        for (LtlPropertyDef g : guarantees) livNames.add(g.lts().name());
        for (LtlPropertyDef a : assumptions) livNames.add(a.lts().name());

        List<LTS> modComponents = new ArrayList<>(components.size());
        for (LTS lts : components) {
            if (livNames.contains(lts.name()) && !lts.isFluent()) {
                lts = new LTS(lts.name(), lts.initialState(), lts.states(),
                        lts.actions(), lts.transitions(), true,
                        lts.acceptingStates(), lts.stateIndex());
            }
            modComponents.add(lts);
        }
        this.components = List.copyOf(modComponents);

        // phases 0..numAssumptions-1 → assumption LTS indices
        // phases numAssumptions..numPhases-1 → guarantee LTS indices
        for (int i = 0; i < numAssumptions; i++) {
            String name = assumptions.get(i).lts().name();
            boolean found = false;
            for (int j = 0; j < this.components.size(); j++) {
                if (this.components.get(j).name().equals(name)) {
                    assumptionIndices[i] = j;
                    phaseComponentIndex[i] = j;
                    found = true; break;
                }
            }
            if (!found)
                throw new IllegalArgumentException("Assumption LTS '" + name + "' not found in components");
        }
        for (int i = 0; i < numGuarantees; i++) {
            String name = guarantees.get(i).lts().name();
            boolean found = false;
            for (int j = 0; j < this.components.size(); j++) {
                if (this.components.get(j).name().equals(name)) {
                    guaranteeIndices[i] = j;
                    phaseComponentIndex[numAssumptions + i] = j;
                    found = true; break;
                }
            }
            if (!found)
                throw new IllegalArgumentException("Guarantee LTS '" + name + "' not found in components");
        }

        // Build per-component transition maps and alphabet.
        compTrans = new ArrayList<>();
        compAlpha = new ArrayList<>();
        Set<String> alpha = new LinkedHashSet<>();
        for (LTS lts : this.components) {
            Map<String, Map<String, String>> actMap = new HashMap<>();
            Set<String> acts = new LinkedHashSet<>(lts.actions());
            for (Transition t : lts.transitions()) {
                actMap.computeIfAbsent(t.action(), k -> new HashMap<>()).put(t.from(), t.to());
                acts.add(t.action());
            }
            compTrans.add(actMap);
            compAlpha.add(acts);
            alpha.addAll(acts);
        }
        alphabet = Collections.unmodifiableSet(alpha);

        // Provide heuristic with a live view of this engine's state.
        List<Set<String>> cm = new ArrayList<>();
        for (int i = 0; i < this.components.size(); i++) cm.add(new LinkedHashSet<>());
        for (int i = 0; i < numGuarantees; i++)
            cm.get(guaranteeIndices[i]).addAll(this.components.get(guaranteeIndices[i]).acceptingStates());
        this.compMarked = Collections.unmodifiableList(cm);
        heuristic.init(new SimpleSynthesisContext(this.components, this.compMarked, this.controllable) {
            @Override public Set<String>              exploredStates()            { return succMap.keySet(); }
            @Override public Set<String>              goals()                     { return goals; }
            @Override public Set<String>              errors()                    { return errors; }
            @Override public List<ExtendedTransition> successorsOf(String s)      { return succMap.getOrDefault(s, List.of()); }
            @Override public Set<String>              predecessorsOf(String s)    { return parents.getOrDefault(s, Set.of()); }
            @Override public String                   plantStateKey(String nodeId) { return plantPart(nodeId); }
            @Override public boolean                  verbose()                   { return OTFDirectedControledSyntesisGR1.this.verbose; }
            @Override public List<ExtendedTransition> getFutureAddToFrontier(ExtendedTransition t) {
                return OTFDirectedControledSyntesisGR1.this.getFutureAddToFrontier(t);
            }
        });

        this.featureCompute = featureCompute;
        if (featureCompute != null) {
            this.featuresCtx = new FeaturesContext(
                    goals, errors, none, succMap, parents,
                    this.controllable, alphabet, this.components, this.compMarked, true);
            featureCompute.init(this.featuresCtx);
        } else {
            this.featuresCtx = null;
        }

        // Expand and classify the initial state.
        s0 = initialState();
        exploreState(s0);

        if (isLosing(s0)) {
            errors.add(s0);
            pending          = new ArrayList<>();
            explorationEnded = true;
        } else {
            none.add(s0);
            if (featuresCtx != null && isMarked(s0)) featuresCtx.markedStateFound = true;
            List<ExtendedTransition> s0Trans = succMap.getOrDefault(s0, List.of());
            for (ExtendedTransition nt : s0Trans) nt.setStep(0);
            pending = new ArrayList<>();
            if (frontierRestriction) {
                Deque<ExtendedTransition> forcedQueue = new ArrayDeque<>();
                distributeTransitions(s0Trans, forcedQueue);
                while (!forcedQueue.isEmpty()) {
                    transitionsExplored++;
                    expandOne(forcedQueue.poll(), forcedQueue);
                }
            } else {
                pending.addAll(s0Trans);
            }
            explorationEnded = false;
        }
    }

    // ── public entry point ────────────────────────────────────────────────────

    /**
     * Runs the otf-dcs-GR(1) algorithm.
     *
     * @return {@link Director#isRealizable()} true iff a GR(1) controller
     *         exists from the initial state.
     */
    public Director run() {
        log("Initial state explored | states=" + succMap.size());

        if (explorationEnded) {
            log("Initial state is losing — UNREALIZABLE");
            Director r = getSynthesisResult();
            heuristic.notifyExplorationEnd(r);
            return r;
        }

        while (!isExplorationEnded()) {
            List<ExtendedTransition> frontier = getFrontier();
            if (frontier.isEmpty()) break;

            int index = heuristic.pick(frontier);
            if (index < 0 || index >= frontier.size()) {
                System.err.println("[DCS-GR1] WARNING: heuristic returned invalid index " + index
                        + " for frontier of size " + frontier.size() + " — using nearest valid index");
                index = Math.max(0, Math.min(index, frontier.size() - 1));
            }
            ExtendedTransition t = frontier.get(index);
            heuristic.printFrontier(frontier, index);

            log("  step " + (transitionsExplored + 1)
                    + " | " + displayNodeId(t.from()) + " --[" + t.action() + "]--> " + displayNodeId(t.to()));

            expand(index);

            if (goals.contains(s0)) {
                log("s0 ∈ Goals — REALIZABLE | states=" + succMap.size()
                        + " transitions=" + transitionsExplored);
                logExplorationSummary();
                Director r = getSynthesisResult();
                heuristic.notifyExplorationEnd(r);
                return r;
            }
            if (explorationEnded) {
                log("s0 ∈ Errors — UNREALIZABLE | states=" + succMap.size()
                        + " transitions=" + transitionsExplored);
                logExplorationSummary();
                Director r = getSynthesisResult();
                heuristic.notifyExplorationEnd(r);
                return r;
            }
        }

        log("Exploration complete | states=" + succMap.size()
                + " transitions=" + transitionsExplored);
        logExplorationSummary();
        Director r = getSynthesisResult();
        heuristic.notifyExplorationEnd(r);
        return r;
    }

    // ── step interface ────────────────────────────────────────────────────────

    public List<ExtendedTransition> getFrontier() {
        prunePending();
        return Collections.unmodifiableList(pending);
    }

    public boolean isExplorationEnded() {
        if (explorationEnded) return true;
        prunePending();
        if (pending.isEmpty()) explorationEnded = true;
        return explorationEnded;
    }

    /**
     * Expands the transition at {@code index} in the current frontier (one DCS step).
     *
     * @param index 0-based index into {@link #getFrontier()}
     */
    public void expand(int index) {
        if (isExplorationEnded())
            throw new IllegalStateException("Exploration has already ended");

        // TODO(perf): swap-remove would be O(1) instead of O(|pending|):
        //   ExtendedTransition t = pending.get(index);
        //   int last = pending.size() - 1;
        //   if (index != last) pending.set(index, pending.get(last));
        //   pending.remove(last);
        // Blocked by DCSFrontierRegressionTest: that test asserts exact frontier element
        // order at each step. Swap-remove reorders the tail, breaking those assertions.
        // To enable: update all frontier snapshots in DCSFrontierRegressionTest to match
        // the new ordering, or change the test to use Set comparison instead of List.
        ExtendedTransition t = pending.remove(index);
        transitionsExplored++;

        if (frontierRestriction) {
            Deque<ExtendedTransition> forcedQueue = new ArrayDeque<>();
            expandOne(t, forcedQueue);
            while (!forcedQueue.isEmpty()) {
                transitionsExplored++;
                expandOne(forcedQueue.poll(), forcedQueue);
            }
        } else {
            expandOne(t, null);
        }

        if (goals.contains(s0) || errors.contains(s0)) explorationEnded = true;
    }

    private void expandOne(ExtendedTransition t, Deque<ExtendedTransition> forcedQueue) {
        int frontierStep = transitionsExplored;
        String e  = t.from();
        String eʹ = t.to();

        if (featuresCtx != null) {
            featuresCtx.lastExpandedFrom = e;
            featuresCtx.lastExpandedTo   = eʹ;
        }

        if (isVisited(eʹ)) {
            addParent(eʹ, e);
            if (errors.contains(eʹ)) {
                propagateError(Set.of(eʹ));
            } else if (goals.contains(eʹ)) {
                propagateGoal(Set.of(eʹ));
            } else {
                Set<String> loop = getMaxLoop(e, eʹ);
                if (!loop.isEmpty()) {
                    if (canBeWinningLoop(loop)) {
                        if (featuresCtx != null) featuresCtx.closedWinningLoopsCount++;
                        Set<String> C = findNewGoalsIn(loop);
                        promoteToGoals(C);
                        propagateGoal(C);
                    } else {
                        Set<String> P = findNewErrorsIn(loop);
                        promoteToErrors(P);
                        propagateError(P);
                    }
                }
            }
        } else {
            exploreState(eʹ);
            addParent(eʹ, e);
            if (isLosing(eʹ)) {
                errors.add(eʹ);
                propagateError(Set.of(eʹ));
            } else {
                none.add(eʹ);
                if (featuresCtx != null && isMarked(eʹ)) featuresCtx.markedStateFound = true;
                List<ExtendedTransition> newTransitions = succMap.getOrDefault(eʹ, List.of());
                for (ExtendedTransition nt : newTransitions) nt.setStep(frontierStep);
                if (frontierRestriction) {
                    distributeTransitions(newTransitions, forcedQueue);
                } else {
                    pending.addAll(newTransitions);
                }
            }
        }
    }

    private void distributeTransitions(List<ExtendedTransition> transitions,
                                       Deque<ExtendedTransition> forcedQueue) {
        List<ExtendedTransition> ctrlTs = new ArrayList<>();
        for (ExtendedTransition nt : transitions) {
            if (controllable.contains(nt.action())) ctrlTs.add(nt);
            else forcedQueue.add(nt);
        }
        if (ctrlTs.size() > 1) {
            pending.addAll(ctrlTs);
        } else if (ctrlTs.size() == 1) {
            forcedQueue.add(ctrlTs.get(0));
        }
    }

    public List<int[]> getFrontierWithFeatures() {
        if (featureCompute == null)
            throw new IllegalStateException(
                    "No FeatureCompute registered — use the constructor overload that accepts FeatureCompute");
        prunePending();
        List<int[]> result = new ArrayList<>(pending.size());
        for (ExtendedTransition tr : pending) result.add(featureCompute.compute(tr));
        return result;
    }

    public List<String> getFeatureNames()    { return featureCompute != null ? featureCompute.getFeatureNames()    : Collections.emptyList(); }
    public String       getFeatureGroupName(){ return featureCompute != null ? featureCompute.getFeatureGroupName() : null; }

    @Override
    public List<ExtendedTransition> getFutureAddToFrontier(ExtendedTransition t) {
        String eʹ = t.to();
        if (isVisited(eʹ)) return List.of();
        exploreState(eʹ);
        if (isLosing(eʹ)) return List.of();
        return List.copyOf(succMap.getOrDefault(eʹ, List.of()));
    }

    public boolean       isRealizable()          { return goals.contains(s0); }
    public int           getStatesExplored()      { return succMap.size(); }
    public int           getTransitionsExplored() { return transitionsExplored; }
    public List<LTS>     getComponents()          { return components; }
    public Set<String>   getControllable()        { return controllable; }
    public Set<String>   getAlphabet()            { return alphabet; }
    public Set<String>                           getGoals()      { return goals; }
    public Set<String>                           getErrors()     { return errors; }
    public Set<String>                           getNone()       { return none; }
    public Map<String, List<ExtendedTransition>> getSuccMap()    { return succMap; }
    public Map<String, Set<String>>              getParents()    { return parents; }
    public List<Set<String>>                     getCompMarked() { return compMarked; }

    public Director getSynthesisResult() {
        if (goals.contains(s0))
            return Director.realizable(buildDirector(), succMap.size(), transitionsExplored);
        Director.TerminationReason reason;
        if (errors.contains(s0))    reason = Director.TerminationReason.ERROR;
        else if (pending.isEmpty()) reason = Director.TerminationReason.FRONTIER_EMPTY;
        else                        reason = Director.TerminationReason.NONE;
        return Director.unrealizable(succMap.size(), transitionsExplored, reason);
    }

    // ── state helpers ─────────────────────────────────────────────────────────

    /** Returns the plant-state portion of a node ID (everything before {@code '#'}). */
    private String plantPart(String nodeId) {
        int h = nodeId.lastIndexOf('#');
        return h >= 0 ? nodeId.substring(0, h) : nodeId;
    }

    /** Returns the phase counter embedded in a node ID (the number after {@code '#'}). */
    private int phaseOf(String nodeId) {
        Integer cached = phaseCache.get(nodeId);
        if (cached != null) return cached;
        int h = nodeId.lastIndexOf('#');
        int phase = h >= 0 ? Integer.parseInt(nodeId.substring(h + 1)) : 0;
        phaseCache.put(nodeId, phase);
        return phase;
    }

    private String initialState() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < components.size(); i++) {
            if (i > 0) sb.append('|');
            sb.append(components.get(i).initialState());
        }
        return sb.append("#0").toString();
    }

    /** Splits the plant-state portion of {@code nodeId} into per-component states. */
    private String[] splitState(String nodeId) {
        String[] cached = splitStateCache.get(nodeId);
        if (cached != null) return cached;
        String[] parts = splitPlantState(plantPart(nodeId));
        splitStateCache.put(nodeId, parts);
        return parts;
    }

    private String[] splitPlantState(String plantState) {
        List<String> parts = new ArrayList<>();
        int start = 0;
        for (int i = 0; i < plantState.length(); i++) {
            if (plantState.charAt(i) == '|') {
                parts.add(plantState.substring(start, i));
                start = i + 1;
            }
        }
        parts.add(plantState.substring(start));
        return parts.toArray(new String[0]);
    }

    /**
     * A state is Büchi-accepting for the controller under the unified-phase rule:
     * <ul>
     *   <li>Assumption phase j &lt; numAssumptions: marked when A_j does NOT hold
     *       (environment violated its liveness obligation → controller wins
     *       vacuously under GR(1) semantics).</li>
     *   <li>Guarantee phase j ≥ numAssumptions: marked when G_{j-numAssumptions}
     *       holds (controller satisfied its current liveness goal).</li>
     * </ul>
     */
    /**
     * Büchi acceptance under GR(1):
     * <ul>
     *   <li>Assumption phase (j &lt; numAssumptions): marked when A_j does NOT hold —
     *       env is failing its obligation, controller wins vacuously.</li>
     *   <li>Guarantee phase (j ≥ numAssumptions): marked when G_{j-numAssumptions} holds,
     *       OR when some assumption A_i is violated AND the controller cannot force A_i
     *       back to ON (no controllable successor makes A_i accepting). The second condition
     *       catches cycles where the environment sustains an assumption violation while the
     *       controller has no controllable actions that could resolve it — giving the
     *       controller a vacuous win without needing to return to assumption phase.</li>
     * </ul>
     */
    public boolean isMarked(String nodeId) {
        if (numPhases == 0) return true;
        String[] parts = splitState(nodeId);
        int j = phaseOf(nodeId);
        int cIdx = phaseComponentIndex[j];
        boolean inAccepting = components.get(cIdx).acceptingStates().contains(parts[cIdx]);
        if (j < numAssumptions) return !inAccepting; // assumption phase
        if (inAccepting) return true;                 // guarantee phase, goal satisfied
        // Guarantee phase, goal not satisfied: check for unresolvable assumption violation.
        for (int i = 0; i < numAssumptions; i++) {
            int aCIdx = assumptionIndices[i];
            if (!components.get(aCIdx).acceptingStates().contains(parts[aCIdx])
                    && !controllerCanSatisfyAssumption(nodeId, aCIdx))
                return true;
        }
        return false;
    }

    /**
     * Returns true if any controllable successor of {@code nodeId} has assumption
     * component {@code assumptionComponentIdx} in its accepting state.
     */
    private boolean controllerCanSatisfyAssumption(String nodeId, int assumptionComponentIdx) {
        if (!succMap.containsKey(nodeId))
            return controllerCanSatisfyAssumptionUncached(nodeId, assumptionComponentIdx);
        String cacheKey = nodeId + '\u0000' + assumptionComponentIdx;
        Boolean cached = assumptionSatCache.get(cacheKey);
        if (cached != null) return cached;
        boolean value = controllerCanSatisfyAssumptionUncached(nodeId, assumptionComponentIdx);
        assumptionSatCache.put(cacheKey, value);
        return value;
    }

    private boolean controllerCanSatisfyAssumptionUncached(String nodeId, int assumptionComponentIdx) {
        for (ExtendedTransition t : succMap.getOrDefault(nodeId, List.of())) {
            if (controllable.contains(t.action())) {
                String[] sp = splitState(t.to());
                if (components.get(assumptionComponentIdx).acceptingStates().contains(sp[assumptionComponentIdx]))
                    return true;
            }
        }
        return false;
    }

    private boolean isIllegal(String s) {
        String[] parts = splitState(s);
        for (int i = 0; i < components.size(); i++) {
            if ("ERROR".equals(parts[i])) return true;
        }
        return false;
    }

    private boolean isLosing(String s) {
        return isIllegal(s) || succMap.getOrDefault(s, List.of()).isEmpty();
    }

    private boolean isVisited(String s) {
        return goals.contains(s) || errors.contains(s) || none.contains(s);
    }

    private void addParent(String child, String parent) {
        parents.computeIfAbsent(child, k -> new LinkedHashSet<>()).add(parent);
    }

    // ── state exploration with phase update ───────────────────────────────────

    /**
     * Computes all outgoing transitions from {@code s}, applying the round-robin
     * phase rule: the phase advances when G_{phase} holds at the <em>source</em>
     * state (departing a marked state).
     */
    private void exploreState(String nodeId) {
        if (succMap.containsKey(nodeId)) return;
        String[] parts = splitState(nodeId);
        splitStateCache.putIfAbsent(nodeId, parts);
        int j = phaseOf(nodeId);
        phaseCache.putIfAbsent(nodeId, j);

        // Phase advances when the current phase's LTS is in its accepting state:
        //   assumption phase j<numAssumptions: advances when A_j holds (env met its obligation)
        //   guarantee phase j>=numAssumptions: advances when G_{j-numAssumptions} holds
        int nextJ = j;
        if (numPhases > 0) {
            int cIdx = phaseComponentIndex[j];
            if (components.get(cIdx).acceptingStates().contains(parts[cIdx]))
                nextJ = (j + 1) % numPhases;
        }

        List<ExtendedTransition> succ = new ArrayList<>();

        for (String action : alphabet) {
            String[] newParts = new String[components.size()];
            boolean valid = true;

            for (int i = 0; i < components.size(); i++) {
                String cur = parts[i];
                if (compAlpha.get(i).contains(action)) {
                    String next = compTrans.get(i).getOrDefault(action, Map.of()).get(cur);
                    if (next != null) {
                        newParts[i] = next;
                    } else if (components.get(i).isFluent()) {
                        newParts[i] = cur;
                    } else {
                        valid = false;
                        break;
                    }
                } else {
                    newParts[i] = cur;
                }
            }
            if (!valid) continue;

            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < components.size(); i++) {
                if (i > 0) sb.append('|');
                sb.append(newParts[i]);
            }
            sb.append('#').append(nextJ); // phase suffix, not a component
            String target = sb.toString();
            phaseCache.putIfAbsent(target, nextJ);
            splitStateCache.putIfAbsent(target, newParts);
            succ.add(new ExtendedTransition(nodeId, action, target));
        }
        succMap.put(nodeId, succ);
    }

    // ── loop analysis ─────────────────────────────────────────────────────────

    private Set<String> getMaxLoop(String loopEnd, String loopStart) {
        Set<String> fwd = bfsForward(loopStart, null);
        Set<String> bwd = bfsBackward(loopStart, fwd);
        fwd.retainAll(bwd);
        fwd.removeIf(s -> errors.contains(s) || goals.contains(s));
        return fwd;
    }

    private Set<String> bfsForward(String start, Set<String> limit) {
        Set<String>   visited = new HashSet<>();
        Queue<String> queue   = new ArrayDeque<>();
        queue.add(start); visited.add(start);
        while (!queue.isEmpty()) {
            for (ExtendedTransition t : succMap.getOrDefault(queue.poll(), List.of())) {
                String tgt = t.to();
                if (!visited.contains(tgt) && (limit == null || limit.contains(tgt))) {
                    visited.add(tgt); queue.add(tgt);
                }
            }
        }
        return visited;
    }

    private Set<String> bfsBackward(String start, Set<String> limit) {
        Set<String>   visited = new HashSet<>();
        Queue<String> queue   = new ArrayDeque<>();
        queue.add(start); visited.add(start);
        while (!queue.isEmpty()) {
            for (String p : parents.getOrDefault(queue.poll(), Set.of())) {
                if (!visited.contains(p) && (limit == null || limit.contains(p))) {
                    visited.add(p); queue.add(p);
                }
            }
        }
        return visited;
    }

    private boolean canBeWinningLoop(Set<String> loop) {
        for (String s : loop) {
            if (isMarked(s)) return true;
            for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
                if (goals.contains(t.to())) return true;
            }
        }
        return false;
    }

    private Set<String> findNewGoalsIn(Set<String> loop) {
        Set<String> C = new LinkedHashSet<>(loop);
        boolean outerChanged = true;
        while (outerChanged) {
            outerChanged = false;
            boolean innerChanged = true;
            while (innerChanged) {
                innerChanged = false;
                Iterator<String> it = C.iterator();
                while (it.hasNext()) {
                    if (!safeInRegion(it.next(), C)) {
                        it.remove(); innerChanged = true; outerChanged = true;
                    }
                }
            }
            Set<String> goalOrMarkedAttr = computeGoalOrMarkedAttractor(C);
            Iterator<String> it = C.iterator();
            while (it.hasNext()) {
                if (!goalOrMarkedAttr.contains(it.next())) {
                    it.remove(); outerChanged = true;
                }
            }
        }
        return C;
    }

    private Set<String> findNewErrorsIn(Set<String> loop) {
        for (String s : loop) {
            for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
                if (!errors.contains(t.to()) && !loop.contains(t.to())) return Set.of();
            }
        }
        return loop;
    }

    private boolean safeInRegion(String s, Set<String> C) {
        boolean unsafeUnc = false, hasInRegion = false;
        for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
            boolean inRegion = C.contains(t.to()) || goals.contains(t.to());
            if (!controllable.contains(t.action())) {
                if (!inRegion) { unsafeUnc = true; break; }
                hasInRegion = true;
            } else {
                if (inRegion) hasInRegion = true;
            }
        }
        return !unsafeUnc && hasInRegion;
    }

    private boolean canReachGoalOrMarkedIn(String s, Set<String> C) {
        return computeGoalOrMarkedAttractor(C).contains(s);
    }

    private Set<String> computeGoalOrMarkedAttractor(Set<String> region) {
        Set<String> targets = new LinkedHashSet<>();
        for (String s : region) {
            if (isMarked(s)) targets.add(s);
        }
        return computeAttractor(targets, region);
    }

    private boolean canReachGoalIn(String s, Set<String> C) {
        return computeAttractor(goals, C).contains(s);
    }

    /**
     * Computes the controllable attractor of {@code targets} within {@code region}.
     *
     * <p>Uses a backward worklist: seed from targets in region, then propagate to
     * predecessors (via the {@code parents} map) that pass the attractor condition.
     * An initial scan handles states adjacent to goals outside the region.
     * O(|region| × |succ|) vs the naive O(|region|² × |succ|) fixpoint.
     */
    private Set<String> computeAttractor(Set<String> targets, Set<String> region) {
        Set<String>   attr = new LinkedHashSet<>();
        Queue<String> work = new ArrayDeque<>();
        for (String s : region) {
            if (targets.contains(s) || goals.contains(s)) { attr.add(s); work.add(s); }
        }
        // One scan to seed states with direct transitions to goals outside region
        // (not discoverable via backward parent edges since goals ∉ region).
        for (String s : region) {
            if (!attr.contains(s) && isAttrCandidate(s, attr)) { attr.add(s); work.add(s); }
        }
        while (!work.isEmpty()) {
            String reached = work.poll();
            for (String p : parents.getOrDefault(reached, Set.of())) {
                if (!attr.contains(p) && region.contains(p) && isAttrCandidate(p, attr)) {
                    attr.add(p); work.add(p);
                }
            }
        }
        return attr;
    }

    private boolean isAttrCandidate(String s, Set<String> attr) {
        boolean allUncInAttr = true, hasInAttr = false;
        for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
            boolean in = attr.contains(t.to()) || goals.contains(t.to());
            if (!controllable.contains(t.action())) {
                if (!in) { allUncInAttr = false; break; }
                hasInAttr = true;
            } else {
                if (in) hasInAttr = true;
            }
        }
        return allUncInAttr && hasInAttr;
    }

    // ── propagation ───────────────────────────────────────────────────────────

    private void propagateGoal(Set<String> newGoals) {
        Set<String> C = ancestorsNone(newGoals);
        boolean changed = true;
        while (changed) {
            changed = false;
            Iterator<String> it = C.iterator();
            while (it.hasNext()) {
                String s = it.next();
                if (!safeInRegion(s, C)) { it.remove(); changed = true; }
            }
            Set<String> goalOrMarkedAttr = computeGoalOrMarkedAttractor(C);
            it = C.iterator();
            while (it.hasNext()) {
                if (!goalOrMarkedAttr.contains(it.next())) { it.remove(); changed = true; }
            }
        }
        promoteToGoals(C);
    }

    private void propagateError(Set<String> newErrors) {
        Set<String>   inQueue = new HashSet<>(ancestorsNone(newErrors));
        Queue<String> work    = new ArrayDeque<>(inQueue);
        while (!work.isEmpty()) {
            String s = work.poll();
            inQueue.remove(s);
            if (!none.contains(s)) continue;
            if (isForcedToError(s)) {
                errors.add(s); none.remove(s); pendingDirty = true;
                for (String p : parents.getOrDefault(s, Set.of())) {
                    if (none.contains(p) && inQueue.add(p)) work.add(p);
                }
            }
        }
    }

    private Set<String> ancestorsNone(Set<String> targets) {
        Set<String>   visited = new HashSet<>();
        Queue<String> queue   = new ArrayDeque<>();
        for (String t : targets) {
            for (String p : parents.getOrDefault(t, Set.of())) {
                if (none.contains(p) && visited.add(p)) queue.add(p);
            }
        }
        while (!queue.isEmpty()) {
            for (String p : parents.getOrDefault(queue.poll(), Set.of())) {
                if (none.contains(p) && visited.add(p)) queue.add(p);
            }
        }
        return visited;
    }

    private void promoteToGoals(Set<String> C) {
        for (String s : C) {
            if (goals.add(s) && none.remove(s)) pendingDirty = true;
        }
    }

    private void promoteToErrors(Set<String> P) {
        for (String s : P) {
            if (errors.add(s) && none.remove(s)) pendingDirty = true;
        }
    }

    private boolean isForcedToError(String s) {
        boolean hasNonError = false;
        for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
            if (!controllable.contains(t.action()) && errors.contains(t.to())) return true;
            if (!errors.contains(t.to())) hasNonError = true;
        }
        return !hasNonError;
    }

    // ── director ──────────────────────────────────────────────────────────────

    private Map<String, List<ExtendedTransition>> buildDirector() {
        Map<String, Set<String>> goalRevAdj = new HashMap<>();
        for (String s : goals) {
            for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
                if (goals.contains(t.to()))
                    goalRevAdj.computeIfAbsent(t.to(), k -> new LinkedHashSet<>()).add(s);
            }
        }

        Map<String, Integer> rank  = new HashMap<>();
        Queue<String>        queue = new ArrayDeque<>();
        for (String s : goals) {
            if (isMarked(s)) { rank.put(s, 0); queue.add(s); }
        }
        while (!queue.isEmpty()) {
            String s = queue.poll();
            int r = rank.get(s);
            for (String p : goalRevAdj.getOrDefault(s, Set.of())) {
                if (!rank.containsKey(p)) { rank.put(p, r + 1); queue.add(p); }
            }
        }

        Map<String, List<ExtendedTransition>> enabled = new HashMap<>();
        for (String s : goals) {
            int minRank = Integer.MAX_VALUE;
            ExtendedTransition bestCtrl = null;
            for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
                if (controllable.contains(t.action()) && goals.contains(t.to())) {
                    int r = rank.getOrDefault(t.to(), Integer.MAX_VALUE);
                    if (r < minRank) { minRank = r; bestCtrl = t; }
                }
            }
            List<ExtendedTransition> trans = new ArrayList<>();
            if (bestCtrl != null) trans.add(bestCtrl);
            for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
                if (!controllable.contains(t.action())) {
                    trans.add(t);
                }
            }
            String key = plantPart(s);
            enabled.merge(key, trans, (a, b) -> {
                for (ExtendedTransition t : b) {
                    if (!controllable.contains(t.action())) a.add(t);
                }
                return a;
            });
        }

        Set<String> reachable = new LinkedHashSet<>();
        Queue<String> bfs = new ArrayDeque<>();
        String s0plant = plantPart(s0);
        if (enabled.containsKey(s0plant)) { reachable.add(s0plant); bfs.add(s0plant); }
        while (!bfs.isEmpty()) {
            String s = bfs.poll();
            for (ExtendedTransition t : enabled.getOrDefault(s, List.of())) {
                String tp = plantPart(t.to());
                if (enabled.containsKey(tp) && !reachable.contains(tp)) {
                    reachable.add(tp);
                    bfs.add(tp);
                }
            }
        }
        enabled.keySet().retainAll(reachable);
        return enabled;
    }

    // ── pending pruning ───────────────────────────────────────────────────────

    private void prunePending() {
        if (!pendingDirty) return;
        pending.removeIf(tr -> !none.contains(tr.from()));
        pendingDirty = false;
    }

    // ── logging ───────────────────────────────────────────────────────────────

    private void logExplorationSummary() {}

    /**
     * Returns the display string for the plant-state portion of {@code nodeId}.
     * When {@link #useNumericIds} is true, each component sub-state is replaced
     * by its integer ID from {@link LTS#stateIndex()}.
     */
    private String displayPlant(String nodeId) {
        if (!useNumericIds) return plantPart(nodeId);
        String[] parts = splitState(nodeId);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < components.size(); i++) {
            if (i > 0) sb.append('|');
            Integer id = components.get(i).stateIndex().get(parts[i]);
            sb.append(id != null ? id : parts[i]);
        }
        return sb.toString();
    }

    /** Like {@link #displayPlant} but appends the {@code #phase} suffix. */
    private String displayNodeId(String nodeId) {
        if (!useNumericIds) return nodeId;
        return displayPlant(nodeId) + "#" + phaseOf(nodeId);
    }

    private void log(String msg) {
        if (verbose) System.out.println("[DCS-GR1] " + msg);
    }
}
