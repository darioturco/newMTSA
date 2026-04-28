package newmtsa.synthesis;

import newmtsa.parser.ast.LTS;
import newmtsa.parser.ast.LtlPropertyDef;
import newmtsa.parser.ast.Transition;
import newmtsa.synthesis.features.FeatureCompute;
import newmtsa.synthesis.features.FeaturesContext;
import newmtsa.synthesis.gr1.OTFDirectedControledSyntesisGR1;
import newmtsa.synthesis.heuristics.Heuristic;
import newmtsa.synthesis.heuristics.SimpleSynthesisContext;

import java.util.*;

/**
 * Step-by-step interface to the DCS algorithm for use from Python (e.g. via JPype).
 * Supports both non-blocking and GR(1) synthesis modes.
 *
 * <p>The Python side drives exploration by repeatedly calling {@link #expand(int)} with
 * the chosen frontier index, making this class suitable as a Reinforcement Learning
 * environment.
 *
 * <p>Use {@link #getSynthesisType()} to determine which mode the instance is running.
 * GR(1) instances delegate internally to {@link OTFDirectedControledSyntesisGR1};
 * non-blocking instances implement the algorithm directly (same logic as
 * {@link newmtsa.synthesis.nonblocking.OTFDirectedControledSyntesisNonBlocking}).
 */
public class DCSForPython {

    /** Whether this instance is running a non-blocking or a GR(1) synthesis. */
    public enum SynthesisType { NON_BLOCKING, GR1 }

    // ── instance identity ─────────────────────────────────────────────────────

    public final String        instanceName;
    public final SynthesisType synthesisType;

    /** GR(1) engine; {@code null} when {@link SynthesisType#NON_BLOCKING}. */
    public final OTFDirectedControledSyntesisGR1 gr1Engine;

    // ── inputs ────────────────────────────────────────────────────────────────

    /** Component list of the parallel composition (including any synthetic fluents). */
    public final List<LTS>   components;
    public final Set<String> controllable;

    /** Heuristic used to guide exploration (publicly readable for Python comparison). */
    public final Heuristic heuristic;

    // ── pre-computed per-component data (non-blocking only) ───────────────────

    private final List<Map<String, Map<String, String>>> compTrans;
    private final List<Set<String>> compAlpha;
    private final Set<String>       alphabet;
    private final List<Set<String>> compMarked;
    private final List<Set<String>> compSafeStates;

    // ── state classification (non-blocking only) ──────────────────────────────

    private final Set<String> goals  = new LinkedHashSet<>();
    private final Set<String> errors = new LinkedHashSet<>();
    private final Set<String> none   = new LinkedHashSet<>();

    // ── exploration structure (non-blocking only) ─────────────────────────────

    private final Map<String, List<ExtendedTransition>> succMap = new HashMap<>();
    private final Map<String, Set<String>>              parents = new HashMap<>();

    // ── step-by-step state (non-blocking only) ────────────────────────────────

    private final String                   s0;
    private       List<ExtendedTransition> pending;
    private       boolean                  explorationEnded;
    private       int                      transitionsExplored = 0;

    // ── feature computation (non-blocking only) ───────────────────────────────

    private final FeatureCompute  featureCompute;
    private final FeaturesContext featuresCtx;

    // ── constructors ──────────────────────────────────────────────────────────

    /**
     * Non-blocking convenience constructor — no feature computation.
     *
     * @param instanceName     human-readable name for this synthesis problem
     * @param components       automata forming the plant (parallel composition)
     * @param safetyProperties LTL safety monitors (may be empty)
     * @param markingActions   event labels that define marked states
     * @param controllable     set of controllable event labels
     * @param heuristic        frontier selection strategy
     */
    public DCSForPython(String               instanceName,
                        List<LTS>            components,
                        List<LtlPropertyDef> safetyProperties,
                        Set<String>          markingActions,
                        Set<String>          controllable,
                        Heuristic            heuristic) {
        this(instanceName, components, safetyProperties, markingActions, controllable, heuristic, null);
    }

    /**
     * Non-blocking full constructor — optionally enables per-transition feature computation.
     *
     * @param featureCompute strategy for computing binary feature vectors; {@code null}
     *                       disables features ({@link #getFrontierWithFeatures()} will throw)
     */
    public DCSForPython(String               instanceName,
                        List<LTS>            components,
                        List<LtlPropertyDef> safetyProperties,
                        Set<String>          markingActions,
                        Set<String>          controllable,
                        Heuristic            heuristic,
                        FeatureCompute       featureCompute) {
        if (controllable.isEmpty())
            throw new IllegalArgumentException("DCS requires at least one controllable action");
        if (markingActions.isEmpty())
            throw new IllegalArgumentException("DCS requires at least one marking action");

        this.instanceName  = instanceName;
        this.synthesisType = SynthesisType.NON_BLOCKING;
        this.gr1Engine     = null;
        this.controllable  = Set.copyOf(controllable);
        this.heuristic     = heuristic;

        compTrans      = new ArrayList<>();
        compAlpha      = new ArrayList<>();
        compMarked     = new ArrayList<>();
        compSafeStates = new ArrayList<>();
        Set<String> alpha = new LinkedHashSet<>();

        List<LTS> allComponents = new ArrayList<>(components);
        for (LtlPropertyDef prop : safetyProperties) {
            LTS monitor = prop.lts();
            if (!monitor.isFluent()) {
                monitor = new LTS(monitor.name(), monitor.initialState(), monitor.states(),
                        monitor.actions(), monitor.transitions(), true,
                        monitor.acceptingStates(), monitor.stateIndex());
            }
            allComponents.add(monitor);
        }

        int numPlantComponents = components.size();
        int numSafetyMonitors  = safetyProperties.size();

        Set<String> plantAlpha = new LinkedHashSet<>();
        for (LTS lts : allComponents) {
            for (Transition t : lts.transitions()) plantAlpha.add(t.action());
        }

        allComponents.add(buildMarkingFluent(markingActions, plantAlpha));
        this.components = List.copyOf(allComponents);

        for (int idx = 0; idx < this.components.size(); idx++) {
            LTS lts = this.components.get(idx);
            boolean isSafetyMonitor = idx >= numPlantComponents
                                   && idx <  numPlantComponents + numSafetyMonitors;
            boolean isMarkingFluent = idx == numPlantComponents + numSafetyMonitors;

            Map<String, Map<String, String>> actMap = new HashMap<>();
            Set<String> acts = new LinkedHashSet<>(lts.actions());
            for (Transition t : lts.transitions()) {
                actMap.computeIfAbsent(t.action(), k -> new HashMap<>())
                      .put(t.from(), t.to());
                acts.add(t.action());
            }
            compTrans.add(actMap);
            compAlpha.add(acts);
            alpha.addAll(acts);

            Set<String> marked = new LinkedHashSet<>();
            if (isMarkingFluent) {
                marked.add("on");
            } else if (!isSafetyMonitor && markingActions.isEmpty()) {
                marked.addAll(lts.acceptingStates());
            }
            compMarked.add(marked);

            Set<String> safe = new LinkedHashSet<>();
            if (isSafetyMonitor) safe.addAll(lts.acceptingStates());
            compSafeStates.add(safe);
        }
        alphabet = Collections.unmodifiableSet(alpha);

        SimpleSynthesisContext hCtx = new SimpleSynthesisContext(
                this.components, compMarked, this.controllable) {
            @Override public Set<String>              exploredStates()       { return succMap.keySet(); }
            @Override public Set<String>              goals()                { return goals; }
            @Override public List<ExtendedTransition> successorsOf(String s) {
                return succMap.getOrDefault(s, List.of());
            }
        };
        heuristic.init(hCtx);

        this.featureCompute = featureCompute;
        if (featureCompute != null) {
            this.featuresCtx = new FeaturesContext(
                    goals, errors, none, succMap, parents,
                    this.controllable, alphabet, this.components, compMarked);
            featureCompute.init(this.featuresCtx);
        } else {
            this.featuresCtx = null;
        }

        s0 = initialState();
        exploreState(s0);

        if (isLosing(s0)) {
            errors.add(s0);
            pending          = new ArrayList<>();
            explorationEnded = true;
        } else {
            none.add(s0);
            if (featuresCtx != null && isMarked(s0)) featuresCtx.markedStateFound = true;
            pending          = new ArrayList<>(succMap.getOrDefault(s0, List.of()));
            explorationEnded = false;
        }
    }

    /**
     * GR(1) constructor — wraps {@link OTFDirectedControledSyntesisGR1} so that both
     * synthesis types share the same Python-facing step-by-step interface.
     *
     * @param instanceName human-readable name for this synthesis problem
     * @param components   plant processes + guarantee/assumption fluents
     * @param assumptions  environment liveness assumptions
     * @param guarantees   controller liveness guarantees G_0 .. G_{k-1}
     * @param controllable set of controllable event labels
     * @param heuristic    frontier selection strategy
     */
    public DCSForPython(String               instanceName,
                        List<LTS>            components,
                        List<LtlPropertyDef> assumptions,
                        List<LtlPropertyDef> guarantees,
                        Set<String>          controllable,
                        Heuristic            heuristic) {
        if (controllable.isEmpty())
            throw new IllegalArgumentException("DCS requires at least one controllable action");

        this.instanceName  = instanceName;
        this.synthesisType = SynthesisType.GR1;
        this.heuristic     = heuristic;
        this.gr1Engine     = new OTFDirectedControledSyntesisGR1(
                components, assumptions, guarantees, controllable, heuristic, false);

        // Expose the engine's (possibly modified) component list and alphabet.
        this.components   = gr1Engine.getComponents();
        this.controllable = gr1Engine.getControllable();
        this.alphabet     = gr1Engine.getAlphabet();

        // Non-blocking-internal structures — not used in GR1 mode.
        this.compTrans      = null;
        this.compAlpha      = null;
        this.compMarked     = null;
        this.compSafeStates = null;
        this.featureCompute = null;
        this.featuresCtx    = null;
        this.s0             = null;
        this.pending        = null;
    }

    // ── public API ────────────────────────────────────────────────────────────

    /** Returns whether this instance is running a non-blocking or a GR(1) synthesis. */
    public SynthesisType getSynthesisType() {
        return synthesisType;
    }

    /** Returns the name of this synthesis instance. */
    public String getInstanceName() {
        return instanceName;
    }

    /** Returns the full action alphabet of the parallel composition. */
    public Set<String> getAlphabet() {
        return alphabet;
    }

    /** Returns the component list used in the parallel composition. */
    public List<LTS> getComponents() {
        return components;
    }

    /**
     * Returns the current frontier: the list of pending transitions (from a None state)
     * that have not yet been expanded. Transitions whose source state has been classified
     * since they were added are pruned first.
     */
    public List<ExtendedTransition> getFrontier() {
        if (synthesisType == SynthesisType.GR1) return gr1Engine.getFrontier();
        prunePending();
        return Collections.unmodifiableList(pending);
    }

    /**
     * Returns {@code true} when no further expansion is possible — either the initial
     * state has been classified (realizable / unrealizable) or the frontier is empty.
     */
    public boolean isExplorationEnded() {
        if (synthesisType == SynthesisType.GR1) return gr1Engine.isExplorationEnded();
        if (explorationEnded) return true;
        prunePending();
        if (pending.isEmpty()) explorationEnded = true;
        return explorationEnded;
    }

    /**
     * Performs one step of the DCS main loop by expanding the transition at position
     * {@code index} in the current (pruned) frontier.
     *
     * @param index index into {@link #getFrontier()} (0-based)
     * @throws IllegalStateException     if exploration has already ended
     * @throws IndexOutOfBoundsException if {@code index} is out of range
     */
    public void expand(int index) {
        if (synthesisType == SynthesisType.GR1) { gr1Engine.expand(index); return; }

        if (isExplorationEnded())
            throw new IllegalStateException("Exploration has already ended");

        ExtendedTransition t = pending.remove(index);
        transitionsExplored++;

        String e  = t.from();
        String eʹ = t.to();

        if (featuresCtx != null) {
            featuresCtx.lastExpandedFrom = e;
            featuresCtx.lastExpandedTo   = eʹ;
        }

        if (isVisited(eʹ)) {
            addParent(eʹ, e);
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
        } else {
            exploreState(eʹ);
            addParent(eʹ, e);
            if (isLosing(eʹ)) {
                errors.add(eʹ);
                propagateError(Set.of(eʹ));
            } else {
                none.add(eʹ);
                if (featuresCtx != null && isMarked(eʹ)) featuresCtx.markedStateFound = true;
                pending.addAll(succMap.getOrDefault(eʹ, List.of()));
            }
        }

        if (goals.contains(s0) || errors.contains(s0)) {
            explorationEnded = true;
        }
    }

    /**
     * Returns one binary feature vector per frontier transition.
     *
     * <p>Only available in {@link SynthesisType#NON_BLOCKING} mode. Requires a
     * {@link FeatureCompute} to have been passed at construction time. The returned list
     * is parallel to {@link #getFrontier()}: element {@code i} is the feature vector for
     * frontier transition {@code i}.
     *
     * @throws UnsupportedOperationException if called in GR(1) mode
     * @throws IllegalStateException         if no {@link FeatureCompute} was provided
     */
    public List<int[]> getFrontierWithFeatures() {
        if (synthesisType == SynthesisType.GR1)
            throw new UnsupportedOperationException(
                    "getFrontierWithFeatures() is not supported in GR(1) mode");
        if (featureCompute == null)
            throw new IllegalStateException(
                    "No FeatureCompute registered — use the constructor overload that accepts FeatureCompute");
        prunePending();
        List<int[]> result = new ArrayList<>(pending.size());
        for (ExtendedTransition t : pending) {
            result.add(featureCompute.compute(t));
        }
        return result;
    }

    /**
     * Returns the synthesis result. Should only be called after
     * {@link #isExplorationEnded()} returns {@code true}.
     */
    public SynthesisResult getSynthesisResult() {
        if (synthesisType == SynthesisType.GR1) return gr1Engine.getSynthesisResult();
        if (goals.contains(s0))
            return SynthesisResult.of(buildDirector(), succMap.size(), transitionsExplored);
        return SynthesisResult.unrealizable(succMap.size(), transitionsExplored);
    }

    // ── additional observable state ───────────────────────────────────────────

    /** Returns {@code true} iff the initial state has been classified as realizable. */
    public boolean isRealizable() {
        if (synthesisType == SynthesisType.GR1) return gr1Engine.isRealizable();
        return goals.contains(s0);
    }

    /** Number of transitions expanded so far. */
    public int getTransitionsExplored() {
        if (synthesisType == SynthesisType.GR1) return gr1Engine.getTransitionsExplored();
        return transitionsExplored;
    }

    /** Number of composite states discovered so far. */
    public int getStatesExplored() {
        if (synthesisType == SynthesisType.GR1) return gr1Engine.getStatesExplored();
        return succMap.size();
    }

    // ── private helpers (non-blocking only) ──────────────────────────────────

    private void prunePending() {
        pending.removeIf(tr -> !none.contains(tr.from()));
    }

    private static LTS buildMarkingFluent(Set<String> markingActions, Set<String> plantAlpha) {
        List<Transition> transitions = new ArrayList<>();
        for (String action : plantAlpha) {
            String target = markingActions.contains(action) ? "on" : "off";
            transitions.add(new Transition("off", action, target));
            transitions.add(new Transition("on",  action, target));
        }
        return new LTS("_marking_fluent", "off",
                       List.of("off", "on"), new ArrayList<>(plantAlpha),
                       transitions, true, Set.of("on"),
                       LTS.buildIndex(List.of("off", "on"), "off"));
    }

    private String initialState() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < components.size(); i++) {
            if (i > 0) sb.append('|');
            sb.append(components.get(i).initialState());
        }
        return sb.toString();
    }

    private String[] splitState(String s) {
        return s.split("\\|", components.size());
    }

    private boolean isVisited(String s) {
        return goals.contains(s) || errors.contains(s) || none.contains(s);
    }

    private void addParent(String child, String parent) {
        parents.computeIfAbsent(child, k -> new LinkedHashSet<>()).add(parent);
    }

    private boolean isMarked(String s) {
        String[] parts = splitState(s);
        for (int i = 0; i < components.size(); i++) {
            Set<String> marked = compMarked.get(i);
            if (!marked.isEmpty() && !marked.contains(parts[i])) return false;
        }
        return true;
    }

    private boolean isIllegal(String s) {
        String[] parts = splitState(s);
        for (int i = 0; i < components.size(); i++) {
            if ("ERROR".equals(parts[i])) return true;
            Set<String> safe = compSafeStates.get(i);
            if (!safe.isEmpty() && !safe.contains(parts[i])) return true;
        }
        return false;
    }

    private boolean isLosing(String s) {
        if (isIllegal(s)) return true;
        return succMap.getOrDefault(s, List.of()).isEmpty();
    }

    private void exploreState(String s) {
        if (succMap.containsKey(s)) return;
        String[] parts = splitState(s);
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
            succ.add(new ExtendedTransition(s, action, sb.toString()));
        }
        succMap.put(s, succ);
    }

    private Set<String> getMaxLoop(String loopEnd, String loopStart) {
        Set<String> forward  = bfsForward(loopStart, null);
        Set<String> backward = bfsBackward(loopStart, forward);
        forward.retainAll(backward);
        forward.removeIf(s -> errors.contains(s) || goals.contains(s));
        return forward;
    }

    private Set<String> bfsForward(String start, Set<String> limit) {
        Set<String>   visited = new LinkedHashSet<>();
        Queue<String> queue   = new ArrayDeque<>();
        queue.add(start); visited.add(start);
        while (!queue.isEmpty()) {
            String s = queue.poll();
            for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
                String tgt = t.to();
                if (!visited.contains(tgt) && (limit == null || limit.contains(tgt))) {
                    visited.add(tgt); queue.add(tgt);
                }
            }
        }
        return visited;
    }

    private Set<String> bfsBackward(String start, Set<String> limit) {
        Set<String>   visited = new LinkedHashSet<>();
        Queue<String> queue   = new ArrayDeque<>();
        queue.add(start); visited.add(start);
        while (!queue.isEmpty()) {
            String s = queue.poll();
            for (String p : parents.getOrDefault(s, Set.of())) {
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
                    String s = it.next();
                    if (!safeInRegion(s, C)) {
                        it.remove();
                        innerChanged = true;
                        outerChanged = true;
                    }
                }
            }
            Iterator<String> it = C.iterator();
            while (it.hasNext()) {
                String s = it.next();
                if (!canReachGoalOrMarkedIn(s, C)) {
                    it.remove();
                    outerChanged = true;
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
        boolean unsafeUnc   = false;
        boolean hasInRegion = false;
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
        Set<String>   visited = new LinkedHashSet<>();
        Queue<String> queue   = new ArrayDeque<>();
        queue.add(s); visited.add(s);
        while (!queue.isEmpty()) {
            String cur = queue.poll();
            if (goals.contains(cur)) return true;
            if (isMarked(cur)) return true;
            for (ExtendedTransition t : succMap.getOrDefault(cur, List.of())) {
                String tgt = t.to();
                if (goals.contains(tgt)) return true;
                if (C.contains(tgt) && !visited.contains(tgt)) {
                    visited.add(tgt); queue.add(tgt);
                }
            }
        }
        return false;
    }

    private boolean canReachGoalIn(String s, Set<String> C) {
        Set<String>   visited = new LinkedHashSet<>();
        Queue<String> queue   = new ArrayDeque<>();
        queue.add(s); visited.add(s);
        while (!queue.isEmpty()) {
            String cur = queue.poll();
            for (ExtendedTransition t : succMap.getOrDefault(cur, List.of())) {
                String tgt = t.to();
                if (goals.contains(tgt)) return true;
                if (C.contains(tgt) && !visited.contains(tgt)) {
                    visited.add(tgt); queue.add(tgt);
                }
            }
        }
        return false;
    }

    private void propagateGoal(Set<String> newGoals) {
        Set<String> C = ancestorsNone(newGoals);
        boolean changed = true;
        while (changed) {
            changed = false;
            Iterator<String> it = C.iterator();
            while (it.hasNext()) {
                String s = it.next();
                if (!safeInRegion(s, C) || !canReachGoalIn(s, C)) {
                    it.remove();
                    changed = true;
                }
            }
        }
        promoteToGoals(C);
    }

    private void propagateError(Set<String> newErrors) {
        Set<String> C = ancestorsNone(newErrors);
        boolean changed = true;
        while (changed) {
            changed = false;
            Iterator<String> it = C.iterator();
            while (it.hasNext()) {
                String s = it.next();
                if (isForcedToError(s)) {
                    errors.add(s); none.remove(s);
                    it.remove();
                    changed = true;
                }
            }
        }
    }

    private Set<String> ancestorsNone(Set<String> targets) {
        Set<String>   visited = new LinkedHashSet<>();
        Queue<String> queue   = new ArrayDeque<>();
        for (String t : targets) {
            for (String p : parents.getOrDefault(t, Set.of())) {
                if (none.contains(p) && !visited.contains(p)) {
                    visited.add(p); queue.add(p);
                }
            }
        }
        while (!queue.isEmpty()) {
            String s = queue.poll();
            for (String p : parents.getOrDefault(s, Set.of())) {
                if (none.contains(p) && !visited.contains(p)) {
                    visited.add(p); queue.add(p);
                }
            }
        }
        return visited;
    }

    private void promoteToGoals(Set<String> C) {
        for (String s : C) {
            if (!goals.contains(s)) { goals.add(s); none.remove(s); }
        }
    }

    private void promoteToErrors(Set<String> P) {
        for (String s : P) {
            if (!errors.contains(s)) { errors.add(s); none.remove(s); }
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

    private Director buildDirector() {
        Map<String, Set<String>> goalRevAdj = new HashMap<>();
        for (String s : goals) {
            for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
                if (goals.contains(t.to())) {
                    goalRevAdj.computeIfAbsent(t.to(), k -> new LinkedHashSet<>()).add(s);
                }
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

        Map<String, Set<String>> enabled = new HashMap<>();
        for (String s : goals) {
            int minRank = Integer.MAX_VALUE;
            for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
                if (controllable.contains(t.action()) && goals.contains(t.to())) {
                    int r = rank.getOrDefault(t.to(), Integer.MAX_VALUE);
                    if (r < minRank) minRank = r;
                }
            }
            Set<String> enabledActions = new LinkedHashSet<>();
            for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
                if (controllable.contains(t.action()) && goals.contains(t.to())
                        && rank.getOrDefault(t.to(), Integer.MAX_VALUE) == minRank) {
                    enabledActions.add(t.action());
                }
            }
            enabled.put(s, enabledActions);
        }
        return new Director(enabled);
    }
}
