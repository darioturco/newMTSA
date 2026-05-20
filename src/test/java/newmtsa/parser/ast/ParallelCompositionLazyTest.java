package newmtsa.parser.ast;

import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class ParallelCompositionLazyTest {

    // A = s0 -a-> s1
    private static LTS ltsA() {
        List<String> states = List.of("s0", "s1");
        return new LTS("A", "s0", states, List.of("a"),
                List.of(new Transition("s0", "a", "s1")),
                false, Collections.emptySet(), LTS.buildIndex(states, "s0"));
    }

    // B = q0 -b-> q1  (disjoint alphabet)
    private static LTS ltsB() {
        List<String> states = List.of("q0", "q1");
        return new LTS("B", "q0", states, List.of("b"),
                List.of(new Transition("q0", "b", "q1")),
                false, Collections.emptySet(), LTS.buildIndex(states, "q0"));
    }

    // C = r0 -a-> r1  (shared action 'a' with A)
    private static LTS ltsC() {
        List<String> states = List.of("r0", "r1");
        return new LTS("C", "r0", states, List.of("a"),
                List.of(new Transition("r0", "a", "r1")),
                false, Collections.emptySet(), LTS.buildIndex(states, "r0"));
    }

    // D = d0 -a-> d1, accepting={d1}
    private static LTS ltsD() {
        List<String> states = List.of("d0", "d1");
        return new LTS("D", "d0", states, List.of("a"),
                List.of(new Transition("d0", "a", "d1")),
                false, Set.of("d1"), LTS.buildIndex(states, "d0"));
    }

    // E = e0 -a-> e1, accepting={e1}
    private static LTS ltsE() {
        List<String> states = List.of("e0", "e1");
        return new LTS("E", "e0", states, List.of("a"),
                List.of(new Transition("e0", "a", "e1")),
                false, Set.of("e1"), LTS.buildIndex(states, "e0"));
    }

    @Test
    void singleComponent_returnsSameStructureWithComposedName() {
        LTS result = new ParallelCompositionLazy("Composed", List.of(ltsA())).execute();

        assertEquals("Composed", result.name());
        assertEquals("s0", result.initialState());
        assertEquals(2, result.states().size());
        assertEquals(1, result.transitions().size());
    }

    @Test
    void disjointAlphabets_interleave_fourStates() {
        LTS result = new ParallelCompositionLazy("AB", List.of(ltsA(), ltsB())).execute();

        // s0|q0, s1|q0, s0|q1, s1|q1 — all reachable
        assertEquals(4, result.states().size());
        assertEquals(4, result.transitions().size());
        assertTrue(result.actions().containsAll(List.of("a", "b")));
    }

    @Test
    void sharedAction_synchronize_twoStates() {
        // A and C share action 'a' — must synchronize, no interleaving
        LTS result = new ParallelCompositionLazy("AC", List.of(ltsA(), ltsC())).execute();

        // only s0|r0 -a-> s1|r1; no other reachable states
        assertEquals(2, result.states().size());
        assertEquals(1, result.transitions().size());
        assertEquals("s0|r0", result.initialState());
        Transition t = result.transitions().get(0);
        assertEquals("a", t.action());
        assertEquals("s0|r0", t.from());
        assertEquals("s1|r1", t.to());
    }

    @Test
    void initialStateHasIndexZero() {
        LTS result = new ParallelCompositionLazy("AB", List.of(ltsA(), ltsB())).execute();

        assertEquals(0, result.stateIndex().get(result.initialState()));
    }

    @Test
    void acceptingStates_pairWhereBoothAccept() {
        // D and E share 'a'; both accept at d1/e1 respectively
        LTS result = new ParallelCompositionLazy("DE", List.of(ltsD(), ltsE())).execute();

        assertEquals(2, result.states().size());
        assertTrue(result.acceptingStates().contains("d1|e1"));
        assertFalse(result.acceptingStates().contains("d0|e0"));
    }
}
