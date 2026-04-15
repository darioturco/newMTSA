package newmtsa.synthesis.nonblocking;

import newmtsa.parser.ast.LTS;
import newmtsa.parser.ast.LtlPropertyDef;
import newmtsa.parser.ast.Transition;
import newmtsa.synthesis.Director;
import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.SynthesisResult;
import newmtsa.synthesis.heuristics.Heuristic;

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
 */
public class OTFDirectedControledSyntesisNonBlocking {

    // ── inputs ────────────────────────────────────────────────────────────────

    private final List<LTS>     components;
    private final Set<String>   controllable;
    private final Heuristic     heuristic;
    private final boolean       verbose;

    // ── pre-computed per-component data ───────────────────────────────────────

    /** action → (from → to) for each component. */
    private final List<Map<String, Map<String, String>>> compTrans;
    /** alphabet of each component. */
    private final List<Set<String>> compAlpha;
    /** full union alphabet. */
    private final Set<String> alphabet;
    /**
     * Marked states per component.  Empty means "all states are marked"
     * (FSP default for processes without explicit marking declarations).
     */
    private final List<Set<String>> compMarked;
    /**
     * Safe states per component for safety properties.  For non-safety components
     * this is empty (no constraint).  For safety monitors: acceptingStates of the
     * safety LTS = states where the property HOLDS.  States not in this set are
     * ILLEGAL — the controller must never reach them.
     * Empty = no safety constraint on this component.
     */
    private final List<Set<String>> compSafeStates;

    // ── state classification ──────────────────────────────────────────────────

    /** W⁺: states from which the controller can guarantee reaching a marked state. */
    private final Set<String> goals  = new LinkedHashSet<>();
    /** W⁻: states from which no non-blocking director exists. */
    private final Set<String> errors = new LinkedHashSet<>();
    /** Not yet classified. */
    private final Set<String> none   = new LinkedHashSet<>();

    // ── exploration structure (ES) ────────────────────────────────────────────

    /** Outgoing transitions in ES for each explored state. */
    private final Map<String, List<ExtendedTransition>> succMap = new HashMap<>();
    /** Reverse adjacency: parent states for backward propagation. */
    private final Map<String, Set<String>>              parents = new HashMap<>();

    // ── stats ─────────────────────────────────────────────────────────────────

    private int transitionsExplored = 0;

    // ── constructor ───────────────────────────────────────────────────────────

    /**
     * @param components       automata that form the plant (parallel composition)
     * @param safetyProperties LTL safety properties (optional, may be empty).
     *                         Each property's {@link LTS#acceptingStates()} contains
     *                         the SAFE states (where the property holds).  States
     *                         not in that set are illegal — the controller must
     *                         prevent reaching them.  These monitors are added to
     *                         the parallel composition and never block events
     *                         (treated as fluents for synchronisation purposes).
     * @param markingActions   event labels that, when fired, lead to a marked state
     *                         in the component that defines them.
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
        if (controllable.isEmpty())
            throw new IllegalArgumentException(
                    "Non-blocking DCS requires at least one controllable action");
        if (markingActions.isEmpty())
            throw new IllegalArgumentException(
                    "Non-blocking DCS requires at least one marking action");

        this.controllable = Set.copyOf(controllable);
        this.heuristic    = heuristic;
        this.verbose      = verbose;

        compTrans      = new ArrayList<>();
        compAlpha      = new ArrayList<>();
        compMarked     = new ArrayList<>();
        compSafeStates = new ArrayList<>();
        Set<String> alpha = new LinkedHashSet<>();

        // Build the combined component list: plant components first, then safety monitors.
        List<LTS> allComponents = new ArrayList<>(components);
        for (LtlPropertyDef prop : safetyProperties) {
            LTS monitor = prop.lts();
            // Safety monitors are treated as fluents: they never block a transition
            // whose event is not in their alphabet; missing transitions self-loop.
            if (!monitor.isFluent()) {
                monitor = new LTS(monitor.name(), monitor.initialState(), monitor.states(),
                        monitor.actions(), monitor.transitions(), true /*isFluent*/,
                        monitor.acceptingStates());
            }
            allComponents.add(monitor);
        }
        this.components = List.copyOf(allComponents);

        for (int idx = 0; idx < this.components.size(); idx++) {
            LTS lts = this.components.get(idx);
            boolean isSafety = idx >= components.size(); // safety monitors come last

            Map<String, Map<String, String>> actMap = new HashMap<>();
            Set<String> acts = new LinkedHashSet<>();
            for (Transition t : lts.transitions()) {
                actMap.computeIfAbsent(t.action(), k -> new HashMap<>())
                      .put(t.from(), t.to());
                acts.add(t.action());
            }
            compTrans.add(actMap);
            compAlpha.add(acts);
            alpha.addAll(acts);

            // Marked states for this component.
            Set<String> marked = new LinkedHashSet<>();
            if (isSafety) {
                // Safety monitors do not contribute to marking — treat all states as marked.
                // (marking is defined only over the plant components)
            } else if (!markingActions.isEmpty()) {
                for (Transition t : lts.transitions()) {
                    if (markingActions.contains(t.action())) marked.add(t.to());
                }
            } else {
                marked.addAll(lts.acceptingStates());
            }
            compMarked.add(marked);

            // Safe states for this component.
            Set<String> safe = new LinkedHashSet<>();
            if (isSafety) {
                // acceptingStates of the safety monitor = states where the property holds.
                // Empty acceptingStates means no constraint (all states safe).
                safe.addAll(lts.acceptingStates());
            }
            // For non-safety components, safe is empty = no safety constraint.
            compSafeStates.add(safe);
        }
        alphabet = Collections.unmodifiableSet(alpha);
    }

    // ── public entry point ────────────────────────────────────────────────────

    /**
     * Runs the otf-dcs algorithm (Listing 1 of Ciolek et al. 2023).
     *
     * @return {@link SynthesisResult#isRealizable()} true iff a non-blocking
     *         director exists from the initial state.
     */
    public SynthesisResult run() {
        String s0 = initialState();
        log("Initial state: " + s0);
        expand(s0);

        // Classify initial state immediately (Alg. lines 5-9).
        if (isLosing(s0)) {
            log("Initial state is losing — UNREALIZABLE");
            errors.add(s0);
            return SynthesisResult.unrealizable(succMap.size(), transitionsExplored);
        }
        if (isWinning(s0)) {
            log("Initial state is winning (terminal marked) — REALIZABLE");
            goals.add(s0);
            return SynthesisResult.of(buildDirector(), succMap.size(), transitionsExplored);
        }
        none.add(s0);

        List<ExtendedTransition> pending = new ArrayList<>(succMap.getOrDefault(s0, List.of()));
        log("Initial pending: " + pending.size() + " transitions");

        // Main loop (Alg. line 10).
        while (!pending.isEmpty()) {
            logFrontier(pending);

            ExtendedTransition t = heuristic.pick(pending);
            pending.remove(t);
            transitionsExplored++;

            String e  = t.from();
            String γ  = t.action();
            String eʹ = t.to();

            log("  step " + transitionsExplored
                    + " | " + e + " --[" + γ + "]--> " + eʹ);

            if (isVisited(eʹ)) {
                // eʹ already in ES — check for new loop (Alg. lines 18-29).
                addParent(eʹ, e);
                Set<String> loop = getMaxLoop(e, eʹ);
                if (!loop.isEmpty()) {
                    log("    loop detected: " + loop.size() + " states");
                    if (canBeWinningLoop(loop)) {
                        Set<String> C = findNewGoalsIn(loop);
                        log("    winning loop → " + C.size() + " new goals");
                        promoteToGoals(C);
                        propagateGoal(C);
                    } else {
                        Set<String> P = findNewErrorsIn(loop);
                        log("    non-winning loop → " + P.size() + " new errors");
                        promoteToErrors(P);
                        propagateError(P);
                    }
                }
            } else {
                // New state: expand and classify (Alg. lines 12-17).
                expand(eʹ);
                addParent(eʹ, e);

                if (isLosing(eʹ)) {
                    log("    " + eʹ + " is losing (deadlock/unmarked)");
                    errors.add(eʹ);
                    propagateError(Set.of(eʹ));
                } else if (isWinning(eʹ)) {
                    log("    " + eʹ + " is winning (terminal marked)");
                    goals.add(eʹ);
                    propagateGoal(Set.of(eʹ));
                } else {
                    none.add(eʹ);
                    pending.addAll(succMap.getOrDefault(eʹ, List.of()));
                }
            }

            if (goals.contains(s0)) {
                log("s0 ∈ Goals — REALIZABLE"
                        + " | states=" + succMap.size()
                        + " transitions=" + transitionsExplored);
                return SynthesisResult.of(buildDirector(), succMap.size(), transitionsExplored);
            }
            if (errors.contains(s0)) {
                log("s0 ∈ Errors — UNREALIZABLE"
                        + " | states=" + succMap.size()
                        + " transitions=" + transitionsExplored);
                return SynthesisResult.unrealizable(succMap.size(), transitionsExplored);
            }
        }

        log("Exploration complete | states=" + succMap.size()
                + " transitions=" + transitionsExplored
                + " goals=" + goals.size()
                + " errors=" + errors.size()
                + " none=" + none.size());

        if (goals.contains(s0))
            return SynthesisResult.of(buildDirector(), succMap.size(), transitionsExplored);
        return SynthesisResult.unrealizable(succMap.size(), transitionsExplored);
    }

    // ── verbose helpers ───────────────────────────────────────────────────────

    private void log(String msg) {
        if (verbose) System.out.println("[DCS-NB] " + msg);
    }

    private void logFrontier(List<ExtendedTransition> pending) {
        if (!verbose) return;
        System.out.println("[DCS-NB] frontier (" + pending.size() + "):");
        for (ExtendedTransition ft : pending)
            System.out.println("        " + ft.from() + " --[" + ft.action() + "]--> " + ft.to());
    }

    // ── state helpers ─────────────────────────────────────────────────────────

    /**
     * Composite initial state: the initial state of every component, joined by '|'.
     */
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

    // ── marking ───────────────────────────────────────────────────────────────

    /**
     * A composite state is marked iff every component with an explicit marking
     * set has its current sub-state inside that set.
     * Components with an empty marking set contribute no constraint (all-marked
     * convention for plain FSP processes).
     */
    private boolean isMarked(String s) {
        String[] parts = splitState(s);
        for (int i = 0; i < components.size(); i++) {
            Set<String> marked = compMarked.get(i);
            if (!marked.isEmpty() && !marked.contains(parts[i])) return false;
        }
        return true;
    }

    // ── safety (SEI) ──────────────────────────────────────────────────────────

    /**
     * A composite state is illegal (∈ SEI) iff any safety monitor component's
     * current sub-state is NOT in that monitor's safe-state set.
     * Components with an empty {@code compSafeStates} entry are unconstrained.
     */
    private boolean isIllegal(String s) {
        String[] parts = splitState(s);
        for (int i = 0; i < components.size(); i++) {
            Set<String> safe = compSafeStates.get(i);
            if (!safe.isEmpty() && !safe.contains(parts[i])) return true;
        }
        return false;
    }

    // ── state classification (Listing 4) ─────────────────────────────────────

    /**
     * isLosing(s) = (E(s) = ∅ ∧ s ∉ ME) ∨ s ∈ SEI
     * A state is losing when it is a deadlock without marking, OR it violates
     * a safety property.
     */
    private boolean isLosing(String s) {
        if (isIllegal(s)) return true;
        List<ExtendedTransition> succ = succMap.getOrDefault(s, List.of());
        return succ.isEmpty() && !isMarked(s);
    }

    /**
     * isWinning(s) = E(s) = ∅ ∧ s ∈ ME ∧ s ∉ SEI
     * Terminal state that is also marked and not illegal → immediately winning.
     */
    private boolean isWinning(String s) {
        if (isIllegal(s)) return false;
        List<ExtendedTransition> succ = succMap.getOrDefault(s, List.of());
        return succ.isEmpty() && isMarked(s);
    }

    // ── expansion ─────────────────────────────────────────────────────────────

    /**
     * Compute all outgoing transitions from composite state {@code s} by
     * synchronizing the component automata according to the parallel composition
     * rules (Definition 5).
     */
    private void expand(String s) {
        if (succMap.containsKey(s)) return;
        String[] parts = splitState(s);
        List<ExtendedTransition> succ = new ArrayList<>();

        for (String action : alphabet) {
            String[] newParts = new String[components.size()];
            boolean valid = true;

            for (int i = 0; i < components.size(); i++) {
                String cur = parts[i];
                if (compAlpha.get(i).contains(action)) {
                    // Component participates in this action.
                    String next = compTrans.get(i).getOrDefault(action, Map.of()).get(cur);
                    if (next != null) {
                        newParts[i] = next;
                    } else if (components.get(i).isFluent()) {
                        // Safety monitor (fluent): missing transition → self-loop.
                        newParts[i] = cur;
                    } else {
                        // Action in alphabet of component but no transition at cur → blocked.
                        valid = false;
                        break;
                    }
                } else {
                    // Component does not participate → self-loop (interleaving).
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

    // ── loop detection (getMaxLoop – Listing 4) ───────────────────────────────

    /**
     * Returns the maximal set of None/unclassified states that form a loop
     * through the newly discovered back-edge {@code loopEnd → loopStart}.
     *
     * <p>Formally: states reachable from {@code loopStart} in ES that can also
     * reach {@code loopStart}, excluding already-classified states (Goals/Errors).
     */
    private Set<String> getMaxLoop(String loopEnd, String loopStart) {
        Set<String> forward  = bfsForward(loopStart, null);
        Set<String> backward = bfsBackward(loopStart, forward);
        forward.retainAll(backward);
        forward.removeIf(s -> errors.contains(s) || goals.contains(s));
        return forward;
    }

    /** BFS forward from {@code start}, optionally restricted to {@code limit}. */
    private Set<String> bfsForward(String start, Set<String> limit) {
        Set<String>    visited = new LinkedHashSet<>();
        Queue<String>  queue   = new ArrayDeque<>();
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

    /** BFS backward from {@code start}, restricted to {@code limit}. */
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

    // ── canBeWinningLoop (Listing 4) ──────────────────────────────────────────

    /**
     * canBeWinningLoop(loop) =
     *   (∃ em ∈ loop . em ∈ ME_S)  ∨  (∃ s ∈ loop . canReachInOneStep(s, Goals))
     *
     * <p>A loop can be winning iff it contains a marked state (the system can
     * terminate in a desired state inside the loop) or it can escape to an
     * already-known winning state in one transition.
     */
    private boolean canBeWinningLoop(Set<String> loop) {
        for (String s : loop) {
            if (isMarked(s)) return true;
            for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
                if (goals.contains(t.to())) return true;
            }
        }
        return false;
    }

    // ── findNewGoalsIn (Listing 3) ────────────────────────────────────────────

    /**
     * Computes the maximal subset of {@code loop} that is winning.
     *
     * <p>The algorithm (Listing 3 of the paper) runs two nested fixpoints:
     * <ol>
     *   <li><b>Safety trim</b> (inner): remove states that cannot be kept safely
     *       inside the region — i.e. some uncontrollable action exits the region,
     *       or no action remains inside.</li>
     *   <li><b>Reachability trim</b> (outer): remove states from which no path
     *       stays within the region and reaches either a marked state or an
     *       already-winning (Goals) state.</li>
     * </ol>
     */
    private Set<String> findNewGoalsIn(Set<String> loop) {
        Set<String> C = new LinkedHashSet<>(loop);

        boolean outerChanged = true;
        while (outerChanged) {
            outerChanged = false;

            // Inner fixpoint: safety trim.
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

            // Reachability trim: remove states that cannot reach goal-or-marked in C.
            Iterator<String> it = C.iterator();
            while (it.hasNext()) {
                String s = it.next();
                if (!canReachGoalOrMarkedIn(s, C)) {
                    it.remove();
                    outerChanged = true;
                }
            }
        }

        log("    findNewGoalsIn: " + C.size() + " / " + loop.size() + " states survive");
        return C;
    }

    // ── findNewErrorsIn (Listing 3) ───────────────────────────────────────────

    /**
     * If any state in {@code loops} has a transition to a state outside
     * (loops ∪ Errors), the loop can escape — return ∅.
     * Otherwise the entire loop is trapped and becomes new Errors.
     */
    private Set<String> findNewErrorsIn(Set<String> loop) {
        for (String s : loop) {
            for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
                if (!errors.contains(t.to()) && !loop.contains(t.to())) {
                    log("    findNewErrorsIn: escape via " + t + " — not all errors");
                    return Set.of();
                }
            }
        }
        log("    findNewErrorsIn: fully trapped → " + loop.size() + " errors");
        return loop;
    }

    // ── region helpers ────────────────────────────────────────────────────────

    /**
     * A state is safe in region {@code C} (can remain under directed control) when:
     * <ul>
     *   <li>No uncontrollable transition exits {@code C ∪ Goals} (the environment
     *       cannot force the system outside the winning region), and</li>
     *   <li>At least one transition stays inside {@code C ∪ Goals} (a move exists).</li>
     * </ul>
     * This is the implementation of ¬forcedTo(s, ES \ (C ∪ Goals), ES).
     */
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

    /**
     * canReachGoalOrMarkedIn(s, C):
     * BFS from {@code s} restricted to states in {@code C}, checking whether we
     * can reach a state that is either in {@code Goals} (directly or in one step)
     * or is marked (s' ∈ ME_S).
     */
    private boolean canReachGoalOrMarkedIn(String s, Set<String> C) {
        Set<String>   visited = new LinkedHashSet<>();
        Queue<String> queue   = new ArrayDeque<>();
        queue.add(s); visited.add(s);

        while (!queue.isEmpty()) {
            String cur = queue.poll();
            // A state in Goals is a target — one-step escape counts.
            if (goals.contains(cur)) return true;
            // A marked state in C is a valid target.
            if (isMarked(cur)) return true;
            for (ExtendedTransition t : succMap.getOrDefault(cur, List.of())) {
                String tgt = t.to();
                // Exit to Goals counts (canReachInOneStep).
                if (goals.contains(tgt)) return true;
                // Stay within C for further search.
                if (C.contains(tgt) && !visited.contains(tgt)) {
                    visited.add(tgt); queue.add(tgt);
                }
            }
        }
        return false;
    }

    /**
     * canReachGoalIn(s, C):
     * BFS from {@code s} restricted to states in {@code C}, checking whether we
     * can reach a state that has a direct transition into {@code Goals}.
     * Used in propagation (where Goals already contains the targets).
     */
    private boolean canReachGoalIn(String s, Set<String> C) {
        Set<String>   visited = new LinkedHashSet<>();
        Queue<String> queue   = new ArrayDeque<>();
        queue.add(s); visited.add(s);

        while (!queue.isEmpty()) {
            String cur = queue.poll();
            for (ExtendedTransition t : succMap.getOrDefault(cur, List.of())) {
                String tgt = t.to();
                // Direct exit to an already-known goal.
                if (goals.contains(tgt)) return true;
                if (C.contains(tgt) && !visited.contains(tgt)) {
                    visited.add(tgt); queue.add(tgt);
                }
            }
        }
        return false;
    }

    // ── propagation (Listing 2) ───────────────────────────────────────────────

    /**
     * Backward-propagate winning status from {@code newGoals} through None states.
     *
     * <p>Finds all None-state ancestors of {@code newGoals}, then repeatedly
     * removes states that cannot be kept safely in the winning region or cannot
     * reach Goals from within the candidate set. The remaining states are promoted
     * to Goals.
     */
    private void propagateGoal(Set<String> newGoals) {
        // ancestorsNone: BFS backward through None states from the new goals.
        Set<String> C = ancestorsNone(newGoals);
        log("  propagateGoal: " + newGoals.size() + " seeds, "
                + C.size() + " None ancestors");

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

    /**
     * Backward-propagate losing status from {@code newErrors} through None states.
     *
     * <p>Finds all None-state ancestors, then repeatedly promotes states that are
     * forced to Errors (all successors already errors) and cannot reach Goals.
     */
    private void propagateError(Set<String> newErrors) {
        Set<String> C = ancestorsNone(newErrors);
        log("  propagateError: " + newErrors.size() + " seeds, "
                + C.size() + " None ancestors");

        boolean changed = true;
        while (changed) {
            changed = false;
            Iterator<String> it = C.iterator();
            while (it.hasNext()) {
                String s = it.next();
                if (allSuccessorsAreErrors(s) && !canReachGoalIn(s, C)) {
                    errors.add(s); none.remove(s);
                    it.remove();
                    changed = true;
                    log("    propagateError: promoted " + s + " → Errors");
                }
            }
        }
    }

    /**
     * ancestorsNone(targets): BFS backward from {@code targets} through None
     * states only (excluding Goals and Errors), returning all such reachable states.
     */
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

    // ── promotion helpers ─────────────────────────────────────────────────────

    private void promoteToGoals(Set<String> C) {
        int added = 0;
        for (String s : C) {
            if (!goals.contains(s)) { goals.add(s); none.remove(s); added++; }
        }
        if (added > 0) log("  promoted " + added + " states → Goals (total=" + goals.size() + ")");
    }

    private void promoteToErrors(Set<String> P) {
        int added = 0;
        for (String s : P) {
            if (!errors.contains(s)) { errors.add(s); none.remove(s); added++; }
        }
        if (added > 0) log("  promoted " + added + " states → Errors (total=" + errors.size() + ")");
    }

    private boolean allSuccessorsAreErrors(String s) {
        List<ExtendedTransition> succ = succMap.getOrDefault(s, List.of());
        if (succ.isEmpty()) return true;
        for (ExtendedTransition t : succ) {
            if (!errors.contains(t.to())) return false;
        }
        return true;
    }

    // ── director construction (Listing 5) ────────────────────────────────────

    /**
     * Builds the director from the winning set (Goals).
     *
     * <p>rankStates (Listing 5): BFS from marked Goals, assigning rank 0 to
     * goals that are marked, rank 1 to their goal predecessors, etc.
     *
     * <p>bestControllable (Listing 5): at each goal state enable the controllable
     * action leading to the goal successor with the lowest rank.
     */
    private Director buildDirector() {
        // Rank states: BFS from (Goals ∩ marked states).
        Map<String, Integer> rank  = new HashMap<>();
        Queue<String>        queue = new ArrayDeque<>();
        for (String s : goals) {
            if (isMarked(s)) { rank.put(s, 0); queue.add(s); }
        }
        while (!queue.isEmpty()) {
            String s = queue.poll();
            int r = rank.get(s);
            for (String p : parents.getOrDefault(s, Set.of())) {
                if (goals.contains(p) && !rank.containsKey(p)) {
                    rank.put(p, r + 1); queue.add(p);
                }
            }
        }

        // bestControllable: for each goal state, enable the controllable action
        // leading to the goal successor with the smallest rank.
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
