package newmtsa.synthesis.heuristics.ra;

import newmtsa.parser.ast.LTS;
import newmtsa.parser.ast.Transition;
import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.heuristics.SimpleSynthesisContext;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link RAHeuristic} estimate computation.
 *
 * <p>Tests are grounded in two worked examples from the thesis:
 * <ul>
 *   <li><b>Example 1</b>: Table 3.1 / Figure 3.2 from Pazos 2024 (Chapter 3) —
 *       plant {C, F}, state c0|f0, M = ∅. Tests the basic 7-case estimate definition.</li>
 *   <li><b>Example 2</b>: Figure 4.1 from Pazos 2024 (Chapter 4) —
 *       plant {F, G}, state f0|g0, M = ∅. Tests the corrected distance formula
 *       (including the path-weight of the action step itself).</li>
 * </ul>
 *
 * <p>Source: Nicolás Pazos, "Ready Abstraction: una heurística para la técnica de
 * síntesis On-The-Fly de directores Non-Blocking", UBA 2023.
 */
class RAHeuristicTest {

    // ── LTS builder helper ────────────────────────────────────────────────────

    /**
     * Builds an LTS from a flat list of transitions.
     * The initial state is the source of the first transition.
     */
    private static LTS lts(String name, String initial, Set<String> marked,
                            Transition... trans) {
        Set<String>  states  = new LinkedHashSet<>();
        List<String> actions = new ArrayList<>();
        states.add(initial);
        for (Transition t : trans) {
            states.add(t.from());
            states.add(t.to());
            if (!actions.contains(t.action())) actions.add(t.action());
        }
        List<String> stateList = new ArrayList<>(states);
        return new LTS(name, initial, stateList, actions, List.of(trans),
                       false, marked, LTS.buildIndex(stateList, initial));
    }

    /** Builds the per-component marked sets expected by RAHeuristic. */
    private static List<Set<String>> markedSets(Set<String>... sets) {
        return List.of(sets);
    }

    // ── Example 1 automata (Table 3.1 / Figure 3.2, Pazos 2024 §3) ───────────
    //
    // Component C:  c0 --s→ c1 (marked).  Alphabet: {s}.
    // Component F:  f0 --p→ f2 (marked)
    //               f0 --s→ f1
    //               f0 --t→ f3   (f3 is a dead end; f2 not reachable from f3)
    //               f2 --s→ f1
    //               f1 --t→ f2
    //               Alphabet: {p, s, t}.  Marked: {f2}.
    //
    // Composite initial state: "c0|f0".  M = ∅ (no explored composite states).
    //
    // Expected estimates (Table 3.1):
    //   p → ⟨(1,2), (1,1)⟩   C: absent→case-6, 1+d(s,c1)=2;  F: p→f2 marked, 1 step
    //   t → ⟨(1,2), (2,∞)⟩   C: absent→case-6, 1+d(s,c1)=2;  F: t→f3, f2 unreachable

    private static final LTS C_TABLE31 = lts("C", "c0", Set.of("c1"),
            new Transition("c0", "s", "c1"));

    private static final LTS F_TABLE31 = lts("F", "f0", Set.of("f2"),
            new Transition("f0", "p", "f2"),
            new Transition("f0", "s", "f1"),
            new Transition("f0", "t", "f3"),
            new Transition("f2", "s", "f1"),
            new Transition("f1", "t", "f2"));

    private RAHeuristic initExample1(RAHeuristic ra) {
        SimpleSynthesisContext ctx = new SimpleSynthesisContext(
                List.of(C_TABLE31, F_TABLE31),
                markedSets(Set.of("c1"), Set.of("f2")),
                Set.of("p", "s", "t"));
        // M = ∅: no explored states, no goals
        ra.init(ctx);
        return ra;
    }

    @Test
    void table31_p_estimate_base() {
        // Table 3.1, Pazos 2024 §3 — event p, M = ∅
        // C: p absent → case 6, best β=s: 1 + d(s,c1) = 1 + 1 = 2 → (1,2)
        // F: p fires f0→f2 (marked) → (1, 1)
        RAHeuristic ra = initExample1(RAHeuristic.base());
        List<EstimateTuple> est = ra.estimateFor("c0|f0", "p");
        assertEquals(2, est.size(), "should have one tuple per component");
        assertEquals(new EstimateTuple(1, 2), est.get(0), "C component of p");
        assertEquals(new EstimateTuple(1, 1), est.get(1), "F component of p");
    }

    @Test
    void table31_t_estimate_base() {
        // Table 3.1, Pazos 2024 §3 — event t, M = ∅
        // C: t absent → case 6, best β=s: 1 + d(s,c1) = 2 → (1,2)
        // F: t fires f0→f3, f2 unreachable from f3 → (2,∞)
        RAHeuristic ra = initExample1(RAHeuristic.base());
        List<EstimateTuple> est = ra.estimateFor("c0|f0", "t");
        assertEquals(2, est.size());
        assertEquals(new EstimateTuple(1, 2), est.get(0), "C component of t");
        assertEquals(EstimateTuple.UNREACHABLE,  est.get(1), "F component of t");
    }

    @Test
    void table31_s_estimate_base() {
        // Not in Table 3.1 but derivable from the same example
        // C: s fires c0→c1 (marked) → (1, 1)
        // F: s fires f0→f1, then f1 --t→ f2 (dist 1) → (1, 2)
        RAHeuristic ra = initExample1(RAHeuristic.base());
        List<EstimateTuple> est = ra.estimateFor("c0|f0", "s");
        assertEquals(2, est.size());
        assertEquals(new EstimateTuple(1, 1), est.get(0), "C component of s");
        assertEquals(new EstimateTuple(1, 2), est.get(1), "F component of s");
    }

    @Test
    void table31_ordering_p_before_t_when_both_controllable() {
        // Table 3.1, Pazos 2024 §3 — both controllable: p should be preferred
        // p sorted desc: [(1,2),(1,1)], t sorted desc: [(2,∞),(1,2)]
        // lex: (1,2) < (2,∞) → p wins
        RAHeuristic ra = initExample1(RAHeuristic.base());
        ExtendedTransition tP = new ExtendedTransition("c0|f0", "p", "c1|f2");
        ExtendedTransition tT = new ExtendedTransition("c0|f0", "t", "c0|f3");
        ExtendedTransition picked = ra.pick(List.of(tT, tP));
        assertSame(tP, picked, "p should be picked before t (closer to marked)");
    }

    @Test
    void table31_uncontrollable_before_controllable() {
        // Uncontrollable transitions always come before controllable regardless of estimate.
        // Only p is controllable here; t is uncontrollable.
        SimpleSynthesisContext ctx = new SimpleSynthesisContext(
                List.of(C_TABLE31, F_TABLE31),
                markedSets(Set.of("c1"), Set.of("f2")),
                Set.of("p"));
        RAHeuristic ra = RAHeuristic.base();
        ra.init(ctx);
        ExtendedTransition tP = new ExtendedTransition("c0|f0", "p", "c1|f2");
        ExtendedTransition tT = new ExtendedTransition("c0|f0", "t", "c0|f3");
        ExtendedTransition picked = ra.pick(List.of(tP, tT));
        assertSame(tT, picked, "uncontrollable t must be picked before controllable p");
    }

    // ── Example 2 automata (Figure 4.1, Pazos 2024 §4) ───────────────────────
    //
    // This example appears in §4.2 "Cálculo de estimaciones" to illustrate the
    // corrected distance formula (including the ERA edge weight of the action step).
    //
    // Component F:  f0 --a→ f1, f1 --a→ f2 (marked).  Alphabet: {a}.
    // Component G:  g0 --d→ g1 (marked), g0 --a→ g2, g1 --a→ g2.  Alphabet: {a, d}.
    //
    // Composite initial state: "f0|g0".  M = ∅.
    //
    // Expected (NEW formula, Pazos 2024 §4.2):
    //   d → ⟨(1,3), (1,1)⟩
    //     F: d absent → case 6, best β=a: 1 + d(a,f2) = 1 + (1+BFS(f1,f2)) = 1+(1+1) = 3
    //     G: d fires g0→g1 (marked) → 1 step → (1,1)
    //
    // The OLD formula (Ciolek, before Pazos corrections) gave ⟨(1,1),(1,0)⟩
    // because it ignored the ERA edge weight (i.e. the action step itself).

    private static final LTS F_FIG41 = lts("F", "f0", Set.of("f2"),
            new Transition("f0", "a", "f1"),
            new Transition("f1", "a", "f2"));

    private static final LTS G_FIG41 = lts("G", "g0", Set.of("g1"),
            new Transition("g0", "d", "g1"),
            new Transition("g0", "a", "g2"),
            new Transition("g1", "a", "g2"));

    private RAHeuristic initExample2(RAHeuristic ra) {
        SimpleSynthesisContext ctx = new SimpleSynthesisContext(
                List.of(F_FIG41, G_FIG41),
                markedSets(Set.of("f2"), Set.of("g1")),
                Set.of("a", "d"));
        ra.init(ctx);
        return ra;
    }

    @Test
    void figure41_d_estimate_corrected_formula() {
        // Figure 4.1, Pazos 2024 §4.2 — corrected estimate for d, M = ∅
        // F: d absent → case 6: 1 + d(a, f2) where d(a,f2) = 1+BFS(f1,f2) = 1+1 = 2 → total 3
        // G: d fires g0→g1 (marked in 1 step) → (1,1)
        RAHeuristic ra = initExample2(RAHeuristic.base());
        List<EstimateTuple> est = ra.estimateFor("f0|g0", "d");
        assertEquals(2, est.size());
        assertEquals(new EstimateTuple(1, 3), est.get(0), "F component of d (new formula)");
        assertEquals(new EstimateTuple(1, 1), est.get(1), "G component of d");
    }

    @Test
    void figure41_a_estimate() {
        // Companion check for action a in Figure 4.1 example, M = ∅
        // F: a fires f0→f1 (not marked), BFS(f1,f2)=1 → (1, 1+1) = (1,2)
        // G: a fires g0→g2 (not marked), g2 has no outgoing transitions → UNREACHABLE
        RAHeuristic ra = initExample2(RAHeuristic.base());
        List<EstimateTuple> est = ra.estimateFor("f0|g0", "a");
        assertEquals(2, est.size());
        assertEquals(new EstimateTuple(1, 2), est.get(0), "F component of a");
        assertEquals(EstimateTuple.UNREACHABLE,  est.get(1), "G component of a (g2 is dead end)");
    }

    // ── ComponentData unit tests ──────────────────────────────────────────────

    @Test
    void componentData_distToMarked_direct() {
        // Single-component LTS: s0 --a→ s1 --b→ s2 (marked s2)
        // BFS distances to s2: s2→0, s1→1, s0→2
        LTS lts = lts("X", "s0", Set.of("s2"),
                new Transition("s0", "a", "s1"),
                new Transition("s1", "b", "s2"));
        ComponentData cd = new ComponentData(lts, Set.of("s2"));
        Map<String, Integer> dist = cd.distToEachMarked.get("s2");
        assertNotNull(dist);
        assertEquals(0, dist.get("s2"));
        assertEquals(1, dist.get("s1"));
        assertEquals(2, dist.get("s0"));
    }

    @Test
    void componentData_stepsToEnableAction() {
        // LTS: s0 --a→ s1, s1 --b→ s2 (b enabled only at s1)
        // stepsToEnableAction[b]: s1→0, s0→1
        LTS lts = lts("X", "s0", Set.of("s2"),
                new Transition("s0", "a", "s1"),
                new Transition("s1", "b", "s2"));
        ComponentData cd = new ComponentData(lts, Set.of("s2"));
        Map<String, Integer> steps = cd.stepsToEnableAction.get("b");
        assertNotNull(steps);
        assertEquals(0, steps.get("s1"), "b enabled at s1, 0 steps");
        assertEquals(1, steps.get("s0"), "s0 needs 1 step (via a) to reach s1 where b is enabled");
        assertNull(steps.get("s2"), "b not enabled at s2 and s2 has no outgoing — s2 cannot reach b");
    }

    @Test
    void componentData_reachableFromState() {
        // LTS: s0 --a→ s1 --b→ s2 --c→ s0 (cycle).
        // reachableFromState is only computed for states that have outgoing edges.
        // s0 and s1 and s2 all have outgoing edges here.
        LTS lts = lts("X", "s0", Set.of("s2"),
                new Transition("s0", "a", "s1"),
                new Transition("s1", "b", "s2"),
                new Transition("s2", "c", "s0"));
        ComponentData cd = new ComponentData(lts, Set.of("s2"));
        assertTrue(cd.reachableFromState.get("s0").containsAll(Set.of("s0", "s1", "s2")),
                "s0 can reach all states");
        assertTrue(cd.reachableFromState.get("s2").containsAll(Set.of("s0", "s1", "s2")),
                "s2 can reach all states via cycle");
    }

    // ── RA.R: cache invalidation on new marked state ──────────────────────────

    @Test
    void raR_invalidates_estimate_when_marked_state_discovered() {
        // Using Example 1 plant.  Initially M = ∅ → m-flag = 1 (unvisited marked).
        // After adding c1|f2 to explored (making c1 and f2 "visited"), RA.R must
        // recompute: m-flag for paths reaching c1 or f2 should become 0 (visited).
        SimpleSynthesisContext ctx = new SimpleSynthesisContext(
                List.of(C_TABLE31, F_TABLE31),
                markedSets(Set.of("c1"), Set.of("f2")),
                Set.of("p", "s", "t"));
        RAHeuristic ra = RAHeuristic.withR();
        ra.init(ctx);

        // Before: m-flag = 1 (unvisited)
        List<EstimateTuple> before = ra.estimateFor("c0|f0", "p");
        assertEquals(1, before.get(1).m(), "F component: marked not yet visited → m=1");

        // Simulate discovering the composite marked state c1|f2
        ctx.exploredStates.add("c1|f2");
        // Force refresh by calling estimateFor again (RA.R clears cache on new marked)
        List<EstimateTuple> after = ra.estimateFor("c0|f0", "p");
        assertEquals(0, after.get(1).m(), "F component: marked now visited → m=0 (RA.R)");
    }

    // ── RA.G: goal states treated as m = -1 ───────────────────────────────────

    @Test
    void raG_uses_goal_states_as_minus1() {
        // Using Example 1 plant.  After adding c1|f2 to goals, the m-flag for paths
        // reaching c1/f2 should be -1 (Goal) when useG is active.
        SimpleSynthesisContext ctx = new SimpleSynthesisContext(
                List.of(C_TABLE31, F_TABLE31),
                markedSets(Set.of("c1"), Set.of("f2")),
                Set.of("p", "s", "t"));
        RAHeuristic ra = RAHeuristic.withERG();
        ra.init(ctx);

        ctx.goals.add("c1|f2");

        List<EstimateTuple> est = ra.estimateFor("c0|f0", "p");
        assertEquals(-1, est.get(1).m(),
                "F component: f2 is in a Goal composite state → m=-1 (RA.G)");
        assertEquals(-1, est.get(0).m(),
                "C component: c1 is in a Goal composite state → m=-1 (RA.G)");
    }
}
