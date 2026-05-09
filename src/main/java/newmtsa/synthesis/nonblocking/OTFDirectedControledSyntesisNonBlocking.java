package newmtsa.synthesis.nonblocking;

import newmtsa.parser.ast.LTS;
import newmtsa.parser.ast.LtlPropertyDef;
import newmtsa.parser.ast.Transition;
import newmtsa.synthesis.Director;
import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.OTFDirectedControlledSynthesis;
import newmtsa.synthesis.SynthesisResult;
import newmtsa.synthesis.features.FeatureCompute;
import newmtsa.synthesis.features.FeaturesContext;
import newmtsa.synthesis.heuristics.Heuristic;
import newmtsa.synthesis.heuristics.SimpleSynthesisContext;

import java.util.*;

/**
 * On-the-fly Directed Controller Synthesis for safe and non-blocking properties.
 *
 * <p>Implements the otf-dcs algorithm from:
 * <blockquote>
 *   Ciolek et al., "On-the-fly informed search of non-blocking directed controllers",
 *   Automatica 147 (2023) 110731.
 * </blockquote>
 *
 * <p>States are classified into three sets maintained as an invariant:
 * <ul>
 *   <li><b>Goals</b>: states from which the controller can guarantee reaching a marked state.</li>
 *   <li><b>Errors</b>: states from which no such controller exists.</li>
 *   <li><b>None</b>: not yet classified.</li>
 * </ul>
 *
 * <p>Use {@link #expand(int)} to drive exploration step-by-step (suitable for RL environments),
 * or {@link #run()} to execute the full algorithm to completion.
 */
public class OTFDirectedControledSyntesisNonBlocking implements OTFDirectedControlledSynthesis {

    // ── inputs ────────────────────────────────────────────────────────────────

    public final List<LTS>   components;
    public final Set<String> controllable;
    public final Heuristic   heuristic;

    // ── run options ───────────────────────────────────────────────────────────

    private final boolean verbose;
    private final int     expansionLimit;
    private final boolean useNumericIds;

    // ── per-component data ────────────────────────────────────────────────────

    private final List<Map<String, Map<String, String>>> compTrans;
    private final List<Set<String>>                      compAlpha;
    private final Set<String>                            alphabet;
    private final List<Set<String>>                      compMarked;
    private final List<Set<String>>                      compSafeStates;

    // ── state classification ──────────────────────────────────────────────────

    private final Set<String> goals  = new LinkedHashSet<>();
    private final Set<String> errors = new LinkedHashSet<>();
    private final Set<String> none   = new LinkedHashSet<>();

    // ── exploration structure ─────────────────────────────────────────────────

    private final Map<String, List<ExtendedTransition>> succMap = new HashMap<>();
    private final Map<String, Set<String>>              parents = new HashMap<>();
    private final Map<String, String[]>                 splitStateCache = new HashMap<>();
    private       boolean                               pendingDirty = false;

    // ── step-by-step state ────────────────────────────────────────────────────

    private final String                   s0;
    private       List<ExtendedTransition> pending;
    private       boolean                  explorationEnded;
    private       int                      transitionsExplored = 0;

    // ── feature computation ───────────────────────────────────────────────────

    private final FeatureCompute  featureCompute;
    private final FeaturesContext featuresCtx;

    // ── constructors ──────────────────────────────────────────────────────────

    public OTFDirectedControledSyntesisNonBlocking(List<LTS>            components,
                                                   List<LtlPropertyDef> safetyProperties,
                                                   Set<String>          markingActions,
                                                   Set<String>          controllable,
                                                   Heuristic            heuristic,
                                                   boolean              verbose) {
        this(components, safetyProperties, markingActions, controllable,
             heuristic, verbose, Integer.MAX_VALUE, false, null);
    }

    public OTFDirectedControledSyntesisNonBlocking(List<LTS>            components,
                                                   List<LtlPropertyDef> safetyProperties,
                                                   Set<String>          markingActions,
                                                   Set<String>          controllable,
                                                   Heuristic            heuristic,
                                                   boolean              verbose,
                                                   int                  expansionLimit) {
        this(components, safetyProperties, markingActions, controllable,
             heuristic, verbose, expansionLimit, false, null);
    }

    public OTFDirectedControledSyntesisNonBlocking(List<LTS>            components,
                                                   List<LtlPropertyDef> safetyProperties,
                                                   Set<String>          markingActions,
                                                   Set<String>          controllable,
                                                   Heuristic            heuristic,
                                                   boolean              verbose,
                                                   int                  expansionLimit,
                                                   boolean              useNumericIds) {
        this(components, safetyProperties, markingActions, controllable,
             heuristic, verbose, expansionLimit, useNumericIds, null);
    }

    public OTFDirectedControledSyntesisNonBlocking(List<LTS>            components,
                                                   List<LtlPropertyDef> safetyProperties,
                                                   Set<String>          markingActions,
                                                   Set<String>          controllable,
                                                   Heuristic            heuristic,
                                                   boolean              verbose,
                                                   int                  expansionLimit,
                                                   boolean              useNumericIds,
                                                   FeatureCompute       featureCompute) {
        if (controllable.isEmpty())
            throw new IllegalArgumentException("DCS requires at least one controllable action");
        if (markingActions.isEmpty())
            throw new IllegalArgumentException("DCS requires at least one marking action");

        this.verbose        = verbose;
        this.expansionLimit = expansionLimit;
        this.useNumericIds  = useNumericIds;
        this.controllable   = Set.copyOf(controllable);
        this.heuristic      = heuristic;

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
            } else if (!isSafetyMonitor) {
                if (markingActions.isEmpty()) {
                    marked.addAll(lts.acceptingStates());
                } else {
                    for (Transition t : lts.transitions()) {
                        if (markingActions.contains(t.action())) marked.add(t.to());
                    }
                }
            }
            compMarked.add(marked);

            Set<String> safe = new LinkedHashSet<>();
            if (isSafetyMonitor) safe.addAll(lts.acceptingStates());
            compSafeStates.add(safe);
        }
        this.alphabet = Collections.unmodifiableSet(alpha);

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

    // ── public step API ───────────────────────────────────────────────────────

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

    public void expand(int index) {
        if (isExplorationEnded())
            throw new IllegalStateException("Exploration has already ended");

        ExtendedTransition t = pending.get(index);
        int _last = pending.size() - 1;
        if (index != _last) pending.set(index, pending.get(_last));
        pending.remove(_last);
        transitionsExplored++;

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
                pending.addAll(succMap.getOrDefault(eʹ, List.of()));
            }
        }

        if (goals.contains(s0) || errors.contains(s0)) {
            explorationEnded = true;
        }
    }

    // ── public getters ────────────────────────────────────────────────────────

    public boolean isRealizable()          { return goals.contains(s0); }
    public int     getStatesExplored()      { return succMap.size(); }
    public int     getTransitionsExplored() { return transitionsExplored; }
    public List<LTS>                             getComponents()   { return components; }
    public Set<String>                           getControllable() { return controllable; }
    public Set<String>                           getAlphabet()     { return alphabet; }
    public Set<String>                           getGoals()        { return goals; }
    public Set<String>                           getErrors()       { return errors; }
    public Set<String>                           getNone()         { return none; }
    public Map<String, List<ExtendedTransition>> getSuccMap()      { return succMap; }
    public Map<String, Set<String>>              getParents()      { return parents; }
    public List<Set<String>>                     getCompMarked()   { return compMarked; }

    public SynthesisResult getSynthesisResult() {
        if (goals.contains(s0))
            return SynthesisResult.of(buildDirector(), succMap.size(), transitionsExplored);
        SynthesisResult.TerminationReason reason;
        if (errors.contains(s0))    reason = SynthesisResult.TerminationReason.ERROR;
        else if (pending.isEmpty()) reason = SynthesisResult.TerminationReason.FRONTIER_EMPTY;
        else                        reason = SynthesisResult.TerminationReason.NONE;
        return SynthesisResult.unrealizable(succMap.size(), transitionsExplored, reason);
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

    public boolean isMarked(String s) {
        String[] parts = splitState(s);
        for (int i = 0; i < components.size(); i++) {
            Set<String> marked = compMarked.get(i);
            if (!marked.isEmpty() && !marked.contains(parts[i])) return false;
        }
        return true;
    }

    // ── full run loop ─────────────────────────────────────────────────────────

    public SynthesisResult run() {
        log("Initial state explored | states=" + getStatesExplored());

        if (isExplorationEnded()) {
            log("Initial state is losing (deadlock or illegal) — UNREALIZABLE");
            return getSynthesisResult();
        }

        while (!isExplorationEnded()) {
            List<ExtendedTransition> frontier = getFrontier();
            if (frontier.isEmpty()) break;

            logFrontier(frontier);

            int index = heuristic.pick(frontier);
            if (index < 0 || index >= frontier.size()) {
                System.err.println("[DCS-NB] WARNING: heuristic returned invalid index " + index
                        + " for frontier of size " + frontier.size() + " — using nearest valid index");
                index = Math.max(0, Math.min(index, frontier.size() - 1));
            }
            ExtendedTransition picked = frontier.get(index);

            log("  step " + (getTransitionsExplored() + 1)
                    + " | " + index + " | " + formatState(picked.from())
                    + " --[" + picked.action() + "]--> " + formatState(picked.to()));

            expand(index);

            if (isRealizable()) {
                log("s0 ∈ Goals — REALIZABLE"
                        + " | states=" + getStatesExplored()
                        + " transitions=" + getTransitionsExplored());
                return getSynthesisResult();
            }
            if (isExplorationEnded()) {
                log("s0 ∈ Errors — UNREALIZABLE"
                        + " | states=" + getStatesExplored()
                        + " transitions=" + getTransitionsExplored());
                return getSynthesisResult();
            }
            if (getTransitionsExplored() >= expansionLimit) {
                log("Budget exhausted (" + expansionLimit + ") — aborting");
                return SynthesisResult.unrealizable(getStatesExplored(), getTransitionsExplored());
            }
        }

        log("Exploration complete"
                + " | states=" + getStatesExplored()
                + " transitions=" + getTransitionsExplored());
        return getSynthesisResult();
    }

    // ── private helpers ───────────────────────────────────────────────────────

    private void prunePending() {
        if (!pendingDirty) return;
        pending.removeIf(tr -> !none.contains(tr.from()));
        pendingDirty = false;
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
        String[] cached = splitStateCache.get(s);
        if (cached != null) return cached;
        int n = components.size();
        String[] parts = new String[n];
        int idx = 0, start = 0;
        for (int i = 0; i < s.length() && idx < n - 1; i++) {
            if (s.charAt(i) == '|') { parts[idx++] = s.substring(start, i); start = i + 1; }
        }
        parts[idx] = s.substring(start);
        splitStateCache.put(s, parts);
        return parts;
    }

    private boolean isVisited(String s) {
        return goals.contains(s) || errors.contains(s) || none.contains(s);
    }

    private void addParent(String child, String parent) {
        parents.computeIfAbsent(child, k -> new LinkedHashSet<>()).add(parent);
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
        splitStateCache.putIfAbsent(s, parts);
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
                if ("ERROR".equals(newParts[i])) { valid = false; break; }
            }
            if (!valid) continue;

            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < components.size(); i++) {
                if (i > 0) sb.append('|');
                sb.append(newParts[i]);
            }
            String target = sb.toString();
            splitStateCache.putIfAbsent(target, newParts);
            succ.add(new ExtendedTransition(s, action, target));
        }
        succMap.put(s, succ);
    }

    private Set<String> getMaxLoop(String loopEnd, String loopStart) {
        Set<String> forward  = bfsForward(loopStart, none);
        Set<String> backward = bfsBackward(loopStart, forward);
        forward.retainAll(backward);
        forward.removeIf(s -> errors.contains(s) || goals.contains(s));
        return forward;
    }

    private Set<String> bfsForward(String start, Set<String> limit) {
        Set<String>   visited = new HashSet<>();
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
        Set<String>   visited = new HashSet<>();
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
                    if (!safeInRegion(it.next(), C)) { it.remove(); innerChanged = true; outerChanged = true; }
                }
            }
            Set<String> reachable = computeForwardReachable(C);
            Iterator<String> it = C.iterator();
            while (it.hasNext()) {
                if (!reachable.contains(it.next())) { it.remove(); outerChanged = true; }
            }
        }
        return C;
    }

    /** Backward BFS from seeds (marked or adjacent to goals) within C. Returns all states that can reach goals/marked within C. */
    private Set<String> computeForwardReachable(Set<String> C) {
        Set<String>   reachable = new HashSet<>();
        Queue<String> queue     = new ArrayDeque<>();
        for (String s : C) {
            if (goals.contains(s) || isMarked(s)) {
                if (reachable.add(s)) queue.add(s);
                continue;
            }
            for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
                if (goals.contains(t.to())) { if (reachable.add(s)) queue.add(s); break; }
            }
        }
        while (!queue.isEmpty()) {
            for (String p : parents.getOrDefault(queue.poll(), Set.of())) {
                if (C.contains(p) && reachable.add(p)) queue.add(p);
            }
        }
        return reachable;
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
        boolean outerChanged = true;
        while (outerChanged) {
            outerChanged = false;
            boolean innerChanged = true;
            while (innerChanged) {
                innerChanged = false;
                Iterator<String> it = C.iterator();
                while (it.hasNext()) {
                    if (!safeInRegion(it.next(), C)) { it.remove(); innerChanged = true; outerChanged = true; }
                }
            }
            Set<String> goalReach = computeGoalReachable(C);
            Iterator<String> it = C.iterator();
            while (it.hasNext()) {
                if (!goalReach.contains(it.next())) { it.remove(); outerChanged = true; }
            }
        }
        promoteToGoals(C);
    }

    /** Backward BFS from states adjacent to goals within C. Returns all states that can reach goals within C. */
    private Set<String> computeGoalReachable(Set<String> C) {
        Set<String>   reachable = new HashSet<>();
        Queue<String> queue     = new ArrayDeque<>();
        for (String s : C) {
            if (goals.contains(s)) { if (reachable.add(s)) queue.add(s); continue; }
            for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
                if (goals.contains(t.to())) { if (reachable.add(s)) queue.add(s); break; }
            }
        }
        while (!queue.isEmpty()) {
            for (String p : parents.getOrDefault(queue.poll(), Set.of())) {
                if (C.contains(p) && reachable.add(p)) queue.add(p);
            }
        }
        return reachable;
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

    private String formatState(String composite) {
        if (!useNumericIds) return composite;
        String[] parts = composite.split("\\|", components.size());
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < parts.length; i++) {
            if (i > 0) sb.append('|');
            Integer id = (i < components.size()) ? components.get(i).stateIndex().get(parts[i]) : null;
            sb.append(id != null ? id : parts[i]);
        }
        return sb.toString();
    }
}
