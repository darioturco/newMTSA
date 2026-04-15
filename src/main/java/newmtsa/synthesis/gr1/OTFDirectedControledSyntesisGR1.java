package newmtsa.synthesis.gr1;

import newmtsa.parser.ast.LTS;
import newmtsa.parser.ast.LtlPropertyDef;
import newmtsa.parser.ast.Transition;
import newmtsa.synthesis.Director;
import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.SynthesisResult;
import newmtsa.synthesis.heuristics.Heuristic;

import java.util.*;

/**
 * OTF DCS algorithm adapted for GR(1) synthesis (with optional assumptions).
 *
 * <p>Extended state: {@code "s1|s2|...|sk|j"} where s1..sk are the current states
 * of each component LTS (processes + guarantee fluents + assumption fluents) and
 * j is the current obligation index (0..n-1).
 *
 * <p>Obligation j advances to (j+1)%n when the newly reached state has guarantee
 * fluent g_j in its accepting state.
 *
 * <p>A loop is winning iff:
 * <ul>
 *   <li>For every j in {0,...,n-1} the loop contains at least one state where g_j
 *       is satisfied (the standard guarantee completion condition), OR</li>
 *   <li>For some assumption a_i, no state in the loop has a_i in its accepting
 *       state (the assumption is violated throughout the loop — vacuous win).</li>
 * </ul>
 *
 * <p>The winning region within a loop is found via a grow-then-trim fixpoint.
 * The grow phase runs to convergence first, then the trim phase runs to
 * convergence; the outer loop repeats until stable. This ordering ensures that
 * mutually-reachable states within a cycle are discovered before any are trimmed.
 */
public class OTFDirectedControledSyntesisGR1 {

    // ── inputs ────────────────────────────────────────────────────────────────

    private final List<LTS> components;
    private final List<Integer> guaranteeIndices;  // index in components for each g_j
    private final int n;                           // number of guarantees
    private final List<Integer> assumptionIndices; // index in components for each a_i
    private final Set<String> controllable;
    private final Heuristic heuristic;
    private final boolean verbose;

    // ── pre-computed per-component transition maps ────────────────────────────

    private final List<Map<String, Map<String, String>>> compTrans; // action → (from → to)
    private final List<Set<String>> compAlpha;
    private final Set<String> alphabet;

    // ── exploration state ─────────────────────────────────────────────────────

    private final Set<String> goals  = new LinkedHashSet<>();
    private final Set<String> errors = new LinkedHashSet<>();
    private final Set<String> none   = new LinkedHashSet<>();

    private final Map<String, List<ExtendedTransition>> succMap = new HashMap<>();
    private final Map<String, Set<String>> parents = new HashMap<>();

    // ── stats ─────────────────────────────────────────────────────────────────

    private int transitionsExplored = 0;

    // ── constructor ───────────────────────────────────────────────────────────

    public OTFDirectedControledSyntesisGR1(List<LTS> components,
                                           List<LtlPropertyDef> assumptions,
                                           List<LtlPropertyDef> guarantees,
                                           Set<String> controllable,
                                           Heuristic heuristic,
                                           boolean verbose) {
        this.components   = List.copyOf(components);
        this.controllable = Set.copyOf(controllable);
        this.heuristic    = heuristic;
        this.verbose      = verbose;
        this.n            = guarantees.size();

        List<Integer> gIdx = new ArrayList<>();
        for (LtlPropertyDef g : guarantees) {
            for (int i = 0; i < components.size(); i++) {
                if (components.get(i).name().equals(g.name())) { gIdx.add(i); break; }
            }
        }
        this.guaranteeIndices = List.copyOf(gIdx);

        List<Integer> aIdx = new ArrayList<>();
        for (LtlPropertyDef a : assumptions) {
            for (int i = 0; i < components.size(); i++) {
                if (components.get(i).name().equals(a.name())) { aIdx.add(i); break; }
            }
        }
        this.assumptionIndices = List.copyOf(aIdx);

        compTrans = new ArrayList<>();
        compAlpha = new ArrayList<>();
        Set<String> alpha = new LinkedHashSet<>();
        for (LTS lts : this.components) {
            Map<String, Map<String, String>> actMap = new HashMap<>();
            Set<String> acts = new LinkedHashSet<>();
            for (Transition t : lts.transitions()) {
                actMap.computeIfAbsent(t.action(), k -> new HashMap<>()).put(t.from(), t.to());
                acts.add(t.action());
            }
            compTrans.add(actMap);
            compAlpha.add(acts);
            alpha.addAll(acts);
        }
        alphabet = Collections.unmodifiableSet(alpha);
    }

    // ── public entry point ────────────────────────────────────────────────────

    public SynthesisResult run() {
        if (n == 0 && assumptionIndices.isEmpty()) {
            log("No guarantees and no assumptions — trivially unrealizable");
            return SynthesisResult.UNREALIZABLE;
        }

        String s0 = initialState();
        log("Initial state: " + s0);
        expand(s0);

        if (isLosing(s0)) {
            log("Initial state is a deadlock — UNREALIZABLE");
            errors.add(s0);
            return SynthesisResult.UNREALIZABLE;
        }
        none.add(s0);

        List<ExtendedTransition> pending = new ArrayList<>(succMap.getOrDefault(s0, List.of()));
        log("Initial pending: " + pending.size() + " transitions");

        while (!pending.isEmpty()) {
            // Fix: encapsulate in a void function
            if (verbose) {
                System.out.println("[DCS] frontier (" + pending.size() + "):");
                for (ExtendedTransition ft : pending) {
                    System.out.println("        " + ft.from() + " --[" + ft.action() + "]--> " + ft.to());
                }
            }

            ExtendedTransition t = heuristic.pick(pending);
            pending.remove(t);
            transitionsExplored++;

            log("  step " + transitionsExplored
                    + " | expanding: " + t.from() + " --[" + t.action() + "]--> " + t.to());

            String tgt = t.to();

            if (isVisited(tgt)) {
                addParent(tgt, t.from());

                Set<String> loop = findLoop(t.from(), tgt);
                if (!loop.isEmpty()) {
                    boolean winning = canBeWinningLoop(loop);
                    log("    loop " + loop.size() + " states"
                            + " → " + (winning ? "WINNING" : "non-winning"));
                    if (winning) {
                        findNewGoalsIn(loop);
                    } else {
                        findNewErrorsIn(loop);
                    }
                }

            } else {
                expand(tgt);
                addParent(tgt, t.from());

                if (isLosing(tgt)) {
                    log("    deadlock → Errors");
                    errors.add(tgt);
                    propagateError(tgt);

                    // Fix: Must check the winning condition also!!!
                } else {
                    none.add(tgt);
                    pending.addAll(succMap.getOrDefault(tgt, List.of()));
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

    // ── verbose helper ────────────────────────────────────────────────────────

    private void log(String msg) {
        if (verbose) System.out.println("[DCS] " + msg);
    }

    // ── state helpers ─────────────────────────────────────────────────────────

    private String initialState() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < components.size(); i++) {
            if (i > 0) sb.append('|');
            sb.append(components.get(i).initialState());
        }
        sb.append('|').append(0);
        return sb.toString();
    }

    private String[] splitState(String ext) {
        return ext.split("\\|", components.size() + 1);
    }

    private int obligation(String[] parts) {
        return Integer.parseInt(parts[components.size()]);
    }

    private boolean isVisited(String s) {
        return goals.contains(s) || errors.contains(s) || none.contains(s);
    }

    private void addParent(String child, String parent) {
        parents.computeIfAbsent(child, k -> new LinkedHashSet<>()).add(parent);
    }

    // ── expansion ─────────────────────────────────────────────────────────────

    private void expand(String s) {
        if (succMap.containsKey(s)) return;
        String[] parts = splitState(s);
        int j = obligation(parts);

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
                        newParts[i] = cur; // fluents never block; missing transition = self-loop
                    } else {
                        valid = false;
                        break;
                    }
                } else {
                    newParts[i] = cur; // self-loop for non-participating components
                }
            }
            if (!valid) continue;

            // Advance obligation when g_j is satisfied at the new state
            int newJ = j;
            if (!guaranteeIndices.isEmpty()) {
                int gIdx = guaranteeIndices.get(j);
                if (components.get(gIdx).acceptingStates().contains(newParts[gIdx])) {
                    newJ = (j + 1) % n;
                }
            }

            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < components.size(); i++) {
                if (i > 0) sb.append('|');
                sb.append(newParts[i]);
            }
            sb.append('|').append(newJ);
            succ.add(new ExtendedTransition(s, action, sb.toString()));
        }
        succMap.put(s, succ);
    }

    // ── classification ────────────────────────────────────────────────────────

    /** A deadlock (no outgoing transitions) is always losing for GR(1). */
    private boolean isLosing(String s) {
        // Fix: Must also check the accomplish of the guaranties if the assumption is false in this state.
        return succMap.getOrDefault(s, List.of()).isEmpty();
    }

    // ── loop detection ────────────────────────────────────────────────────────

    private Set<String> findLoop(String loopEnd, String loopStart) {
        Set<String> forward  = bfsForward(loopStart, null);
        Set<String> backward = bfsBackward(loopStart, forward);
        forward.retainAll(backward);
        forward.removeIf(s -> errors.contains(s) || goals.contains(s));
        return forward;
    }

    private Set<String> bfsForward(String start, Set<String> limit) {
        Set<String> visited = new LinkedHashSet<>();
        Queue<String> queue = new ArrayDeque<>();
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
        Set<String> visited = new LinkedHashSet<>();
        Queue<String> queue = new ArrayDeque<>();
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

    // ── winning loop check ────────────────────────────────────────────────────

    /**
     * A loop is potentially winning iff:
     * <ul>
     *   <li>For some assumption a_i the controller can enforce a sub-loop where a_i is
     *       never satisfied — the environment fails its liveness obligation so the GR(1)
     *       spec is vacuously true (see {@link #findAssumptionViolationTrap}), OR</li>
     *   <li>For every obligation j the loop contains at least one transition (from a
     *       state with obligation j) where g_j is satisfied at the target, meaning the
     *       obligation index can advance.</li>
     * </ul>
     */
    private boolean canBeWinningLoop(Set<String> loop) {
        // Assumption-violation check: controller can enforce a sub-loop where
        // some assumption is never satisfied.
        for (int aIdx : assumptionIndices) {
            if (!findAssumptionViolationTrap(loop, aIdx).isEmpty()) return true;
        }

        // Guarantee check: for every obligation j, there must exist at least one
        // loop-internal (or goals-exiting) transition from a j-state that satisfies g_j.
        for (int j = 0; j < n; j++) {
            int gIdx = guaranteeIndices.get(j);
            boolean hasAdvance = false;
            outer:
            for (String s : loop) {
                if (obligation(splitState(s)) != j) continue;
                for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
                    if (!loop.contains(t.to()) && !goals.contains(t.to())) continue;
                    String[] toParts = splitState(t.to());
                    if (components.get(gIdx).acceptingStates().contains(toParts[gIdx])) {
                        hasAdvance = true;
                        break outer;
                    }
                }
            }
            if (!hasAdvance) return false;
        }
        return true; // all n obligations can advance (n==0 is vacuously true)
    }

    /**
     * Find the maximal subset S ⊆ loop where:
     * <ol>
     *   <li>No state in S has assumption {@code aIdx} in its accepting state.</li>
     *   <li>No uncontrollable transition from any state in S leaves S (the environment
     *       cannot be forced out of S by uncontrollable actions within the loop).</li>
     *   <li>Every state in S has at least one successor (controllable or uncontrollable)
     *       also in S (the controller can keep the system in S indefinitely).</li>
     * </ol>
     * A non-empty result means the controller can disable appropriate controllable
     * actions to trap the system in S forever, violating assumption {@code aIdx}
     * — making the GR(1) spec vacuously true.
     */
    private Set<String> findAssumptionViolationTrap(Set<String> loop, int aIdx) {
        // Seed: loop states where the assumption is not satisfied.
        Set<String> candidate = new LinkedHashSet<>();
        for (String s : loop) {
            String[] parts = splitState(s);
            if (!components.get(aIdx).acceptingStates().contains(parts[aIdx])) {
                candidate.add(s);
            }
        }
        // Trim to fixpoint: remove states that cannot stay inside the candidate.
        boolean changed = true;
        while (changed) {
            changed = false;
            Iterator<String> it = candidate.iterator();
            while (it.hasNext()) {
                String s = it.next();
                boolean unsafeUnc         = false;
                boolean hasSuccInCandidate = false;
                for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
                    boolean inCand = candidate.contains(t.to());
                    if (!controllable.contains(t.action()) && !inCand) {
                        unsafeUnc = true; // uncontrollable escapes candidate
                    }
                    if (inCand) hasSuccInCandidate = true;
                }
                if (unsafeUnc || !hasSuccInCandidate) {
                    it.remove();
                    changed = true;
                }
            }
        }
        return candidate;
    }

    // ── goal/error propagation ────────────────────────────────────────────────

    /**
     * Find the winning region within a loop.
     *
     * <p>There are two ways a loop can be winning:
     * <ol>
     *   <li><b>Guarantee completion</b>: the obligation cycle completes for every j.
     *       Seed = states where the obligation just advanced. Grow and trim to find
     *       the maximal trap from which the controller can force obligation completion.</li>
     *   <li><b>Assumption violation</b>: for some assumption a_i, no state in the loop
     *       ever satisfies a_i. The winning region is any trap within the loop (the
     *       environment is forever failing its obligation, so the spec is vacuously true).
     *       Seed = entire loop.</li>
     * </ol>
     *
     * <p>The fixpoint uses <b>grow-then-trim</b> ordering: the grow phase runs to
     * convergence before the trim phase runs, preventing premature removal of states
     * that belong together in a cyclic goal region.
     */
    private void findNewGoalsIn(Set<String> loop) {
        Set<String> goalIn = buildSeed(loop);
        log("    findNewGoalsIn: seed=" + goalIn.size() + " / loop=" + loop.size());

        goalIn = growThenTrim(goalIn, loop);

        promoteToGoals(goalIn);
    }

    /**
     * Build the seed set for {@link #findNewGoalsIn}.
     *
     * <p><b>Assumption-violation path</b>: if the controller can enforce a sub-loop
     * where some assumption is never satisfied, that sub-loop trap is the seed.
     * {@link #growThenTrim} may then expand it (e.g. to include states that are
     * uncontrollably forced into the trap and thus also winning).
     *
     * <p><b>Guarantee path</b>: the seed is the set of states that have at least one
     * <em>direct</em> obligation-advancing transition — i.e. a successor (in the loop
     * or already in Goals) where g_j is satisfied, causing the obligation index to
     * advance from j to (j+1)%n.  This is the correct backward-attractor seed: states
     * from which the obligation can advance in a single step, from which we then grow
     * the winning region by adding predecessor states that can force reaching the seed.
     */
    private Set<String> buildSeed(Set<String> loop) {
        // Assumption-violation: use the enforcement trap as seed.
        for (int aIdx : assumptionIndices) {
            Set<String> trap = findAssumptionViolationTrap(loop, aIdx);
            if (!trap.isEmpty()) return trap;
        }

        // Trivially winning when there are no guarantees.
        if (n == 0) return new LinkedHashSet<>(loop);

        // Guarantee path: seed = states from which the controller can force g_j
        // satisfaction in one step regardless of the environment.
        //
        // A state s (with obligation j) is in the seed if:
        //   1. EVERY uncontrollable transition from s satisfies g_j at its target
        //      (the environment cannot "race ahead" and avoid obligation advancement), AND
        //   2. At least one transition (ctrl or unctrl) from s satisfies g_j at a target
        //      that is already in loop ∪ goals (so we make actual progress in the region).
        //
        // Condition 1 is vacuously true when s has no uncontrollable actions.
        Set<String> seed = new LinkedHashSet<>();
        for (String s : loop) {
            String[] parts = splitState(s);
            int j    = obligation(parts);
            int gIdx = guaranteeIndices.get(j);
            boolean allUnctrSatisfy  = true;
            boolean anyInScopeAdvance = false;
            for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
                String tgt    = t.to();
                String[] tParts = splitState(tgt);
                boolean gSat  = components.get(gIdx).acceptingStates().contains(tParts[gIdx]);
                if (!controllable.contains(t.action())) {
                    // Every uncontrollable must satisfy g_j (env cannot block us).
                    if (!gSat) { allUnctrSatisfy = false; break; }
                }
                if (gSat && (loop.contains(tgt) || goals.contains(tgt))) {
                    anyInScopeAdvance = true;
                }
            }
            if (allUnctrSatisfy && anyInScopeAdvance) seed.add(s);
        }
        return seed;
    }

    /**
     * Iterates grow-to-fixpoint then trim-to-fixpoint until stable.
     *
     * <p>Running grow to completion before trimming prevents the ordering artifact
     * where a state is trimmed before its predecessor (which would have made it
     * safe) has been added to the region.
     */
    private Set<String> growThenTrim(Set<String> goalIn, Set<String> loop) {
        boolean outerChanged = true;
        while (outerChanged) {
            outerChanged = false;

            // Grow phase: keep adding states that satisfy the goal-region condition
            boolean grew = true;
            while (grew) {
                grew = false;
                for (String s : loop) {
                    if (!goalIn.contains(s) && inGoalRegion(s, goalIn)) {
                        goalIn.add(s);
                        grew = true;
                        outerChanged = true;
                    }
                }
            }

            // Trim phase: remove states that no longer satisfy the condition
            boolean trimmed = true;
            while (trimmed) {
                trimmed = false;
                Iterator<String> it = goalIn.iterator();
                while (it.hasNext()) {
                    String s = it.next();
                    if (!inGoalRegion(s, goalIn)) {
                        it.remove();
                        trimmed = true;
                        outerChanged = true;
                    }
                }
            }
        }
        return goalIn;
    }

    /** Promote all states in {@code goalIn} to Goals and propagate. */
    private void promoteToGoals(Set<String> goalIn) {
        int before = goals.size();
        for (String s : goalIn) {
            if (!goals.contains(s)) { goals.add(s); none.remove(s); }
        }
        int added = goals.size() - before;
        if (added > 0) {
            log("    findNewGoalsIn: promoted " + added + " states to Goals");
            for (String g : goalIn) propagateGoal(g);
        } else {
            log("    findNewGoalsIn: fixpoint produced no new goals");
        }
    }

    /**
     * A state s can remain in / join GoalIn when:
     * <ul>
     *   <li>No uncontrollable successor escapes GoalIn ∪ Goals.</li>
     *   <li>At least one action stays inside GoalIn ∪ Goals after the controller
     *       disables controllable escapes: either an uncontrollable (always
     *       available, forced inside) or a controllable in GoalIn.</li>
     * </ul>
     */
    private boolean inGoalRegion(String s, Set<String> goalIn) {
        boolean unsafeUnc          = false;
        boolean hasUnc             = false;
        boolean hasCtrlInGoalIn    = false;

        for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
            if (controllable.contains(t.action())) {
                if (goalIn.contains(t.to()) || goals.contains(t.to())) {
                    hasCtrlInGoalIn = true;
                }
            } else {
                hasUnc = true;
                if (!goalIn.contains(t.to()) && !goals.contains(t.to())) {
                    unsafeUnc = true;
                }
            }
        }
        return !unsafeUnc && (hasUnc || hasCtrlInGoalIn);
    }

    private void findNewErrorsIn(Set<String> loop) {
        // Mark loop as Errors only if every state is trapped (no escape outside loop to
        // a non-Error state).
        boolean trapped = true;
        for (String s : loop) {
            for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
                if (!errors.contains(t.to()) && !loop.contains(t.to())) {
                    trapped = false; break;
                }
            }
            if (!trapped) break;
        }
        if (trapped) {
            log("    findNewErrorsIn: loop of " + loop.size() + " is fully trapped → Errors");
            for (String s : loop) { errors.add(s); none.remove(s); }
            for (String s : loop) propagateError(s);
        }
    }

    /**
     * Backward-propagate Goal: a state can be a goal if the controller can force
     * remaining in the goal region — all uncontrollable successors must already be
     * goals, and at least one action remains available.
     *
     * <p>For states that cannot be directly promoted (because some uncontrollable
     * successor is still in {@code none}), we check whether the state is part of
     * a loop of {@code none}-states.  If so, we re-run {@code findNewGoalsIn} on
     * that loop — newly-promoted goals may have made a previously-failing loop
     * winning now.
     */
    private void propagateGoal(String newGoal) {
        Queue<String> queue = new ArrayDeque<>();
        queue.add(newGoal);
        // States that failed direct promotion but may be in a now-winning loop
        Set<String> toCheckLoop = new LinkedHashSet<>();

        while (!queue.isEmpty() || !toCheckLoop.isEmpty()) {
            while (!queue.isEmpty()) {
                String s = queue.poll();
                for (String p : parents.getOrDefault(s, Set.of())) {
                    if (!none.contains(p)) continue;
                    if (canBeGoal(p)) {
                        log("    propagateGoal: promoted " + p);
                        goals.add(p); none.remove(p);
                        queue.add(p);
                        toCheckLoop.remove(p);
                    } else {
                        toCheckLoop.add(p);
                    }
                }
            }

            if (!toCheckLoop.isEmpty()) {
                String candidate = toCheckLoop.iterator().next();
                toCheckLoop.remove(candidate);
                if (!none.contains(candidate)) continue; // already classified

                // Find a loop of none-states that includes this candidate
                Set<String> loop = findNoneLoop(candidate);
                if (!loop.isEmpty() && canBeWinningLoop(loop)) {
                    Set<String> goalIn = buildSeed(loop);
                    goalIn = growThenTrim(goalIn, loop);
                    if (!goalIn.isEmpty()) {
                        int before = goals.size();
                        for (String g : goalIn) {
                            if (!goals.contains(g)) { goals.add(g); none.remove(g); }
                        }
                        if (goals.size() > before) {
                            for (String g : goalIn) queue.add(g);
                        }
                    }
                }
            }
        }
    }

    /**
     * Find a loop of {@code none}-states reachable from and back to {@code start}.
     * Returns the SCC containing {@code start} restricted to {@code none} states,
     * or an empty set if no such loop exists.
     */
    private Set<String> findNoneLoop(String start) {
        // Forward reachable none-states from start
        Set<String> forward = new LinkedHashSet<>();
        Queue<String> q = new ArrayDeque<>();
        q.add(start); forward.add(start);
        while (!q.isEmpty()) {
            String s = q.poll();
            for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
                String tgt = t.to();
                if (none.contains(tgt) && !forward.contains(tgt)) {
                    forward.add(tgt); q.add(tgt);
                }
            }
        }
        // Backward reachable none-states from start (within forward set)
        Set<String> backward = new LinkedHashSet<>();
        q.add(start); backward.add(start);
        while (!q.isEmpty()) {
            String s = q.poll();
            for (String p : parents.getOrDefault(s, Set.of())) {
                if (forward.contains(p) && !backward.contains(p)) {
                    backward.add(p); q.add(p);
                }
            }
        }
        forward.retainAll(backward);
        if (forward.size() <= 1 && !hasSelfLoop(start)) forward.clear();
        return forward;
    }

    private boolean hasSelfLoop(String s) {
        for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
            if (t.to().equals(s)) return true;
        }
        return false;
    }

    /**
     * A state can be promoted to Goals when:
     * <ul>
     *   <li>All uncontrollable successors are in Goals (can't be disabled).</li>
     *   <li>At least one action remains after disabling out-of-Goals controllable
     *       actions: either an uncontrollable successor exists (and is in Goals),
     *       or a controllable successor is in Goals.</li>
     * </ul>
     */
    private boolean canBeGoal(String s) {
        boolean unsafeUncontrollable   = false;
        boolean hasUncontrollable       = false;
        boolean hasControllableInGoals  = false;

        for (ExtendedTransition t : succMap.getOrDefault(s, List.of())) {
            if (controllable.contains(t.action())) {
                if (goals.contains(t.to())) hasControllableInGoals = true;
            } else {
                hasUncontrollable = true;
                if (!goals.contains(t.to())) { unsafeUncontrollable = true; break; }
            }
        }
        return !unsafeUncontrollable && (hasUncontrollable || hasControllableInGoals);
    }

    private void propagateError(String newError) {
        Queue<String> queue = new ArrayDeque<>();
        queue.add(newError);
        while (!queue.isEmpty()) {
            String s = queue.poll();
            for (String p : parents.getOrDefault(s, Set.of())) {
                if (none.contains(p) && allSuccessorsAreErrors(p)) {
                    log("    propagateError: " + p + " → Errors");
                    errors.add(p); none.remove(p); queue.add(p);
                }
            }
        }
    }

    private boolean allSuccessorsAreErrors(String s) {
        List<ExtendedTransition> succ = succMap.getOrDefault(s, List.of());
        if (succ.isEmpty()) return true;
        for (ExtendedTransition t : succ) {
            if (!errors.contains(t.to())) return false;
        }
        return true;
    }

    // ── director construction ─────────────────────────────────────────────────

    private Director buildDirector() {
        // BFS rank: distance from each goal state to the nearest goal-cycle entry
        Map<String, Integer> rank = new HashMap<>();
        Queue<String> queue = new ArrayDeque<>();
        for (String g : goals) { rank.put(g, 0); queue.add(g); }
        while (!queue.isEmpty()) {
            String s = queue.poll();
            int r = rank.get(s);
            for (String p : parents.getOrDefault(s, Set.of())) {
                if (goals.contains(p) && !rank.containsKey(p)) {
                    rank.put(p, r + 1); queue.add(p);
                }
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
