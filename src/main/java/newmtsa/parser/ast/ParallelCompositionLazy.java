package newmtsa.parser.ast;

import java.util.*;

/**
 * A lazy parallel composition of two or more LTS.
 *
 * <p>The {@code ||} operator in FSP computes the parallel composition of multiple
 * LTS, producing a new (potentially much larger) LTS.  Because this operation is
 * expensive, it is deferred: the operands are stored and the composition is only
 * executed when {@link #execute()} is called.
 *
 * <ul>
 *   <li>{@code name}       — name of the resulting composed LTS</li>
 *   <li>{@code components} — the LTS operands in declaration order</li>
 * </ul>
 *
 * <p>Composition semantics (CSP/FSP synchronous parallel):
 * shared actions synchronize; non-shared actions interleave.
 * Product states are pairs "sA|sB" (pipe-separated, matching the existing
 * convention used in assert compound states).  Only reachable states are
 * generated via BFS.  Accepting states are pairs where both components are
 * individually accepting.
 */
public record ParallelCompositionLazy(String name, List<LTS> components) {

    /** Execute the parallel composition and return the resulting LTS. */
    public LTS execute() {
        if (components.isEmpty())
            throw new IllegalStateException(
                    "ParallelCompositionLazy '" + name + "' has no components to compose.");

        LTS result = components.get(0);
        for (int i = 1; i < components.size(); i++)
            result = compose(result, components.get(i));

        return new LTS(name, result.initialState(), result.states(), result.actions(),
                result.transitions(), false, result.acceptingStates(), result.stateIndex());
    }

    private static LTS compose(LTS a, LTS b) {
        Set<String> aAlpha = new HashSet<>(a.actions());
        Set<String> bAlpha = new HashSet<>(b.actions());

        Set<String> shared = new HashSet<>(aAlpha);
        shared.retainAll(bAlpha);

        Map<String, List<Transition>> aMap = transMap(a.transitions());
        Map<String, List<Transition>> bMap = transMap(b.transitions());

        String initState = a.initialState() + "|" + b.initialState();

        List<String> states = new ArrayList<>();
        List<Transition> transitions = new ArrayList<>();
        Set<String> accepting = new HashSet<>();
        Set<String> visited = new LinkedHashSet<>();

        Queue<String[]> queue = new ArrayDeque<>();
        queue.add(new String[]{a.initialState(), b.initialState()});
        visited.add(initState);
        states.add(initState);
        if (a.acceptingStates().contains(a.initialState()) && b.acceptingStates().contains(b.initialState()))
            accepting.add(initState);

        while (!queue.isEmpty()) {
            String[] pair = queue.poll();
            String sA = pair[0], sB = pair[1];
            String cur = sA + "|" + sB;

            List<Transition> aTrans = aMap.getOrDefault(sA, Collections.emptyList());
            List<Transition> bTrans = bMap.getOrDefault(sB, Collections.emptyList());

            // Shared actions: both must step together
            for (Transition ta : aTrans) {
                if (shared.contains(ta.action())) {
                    for (Transition tb : bTrans) {
                        if (tb.action().equals(ta.action())) {
                            String next = ta.to() + "|" + tb.to();
                            transitions.add(new Transition(cur, ta.action(), next));
                            if (visited.add(next)) {
                                states.add(next);
                                queue.add(new String[]{ta.to(), tb.to()});
                                if (a.acceptingStates().contains(ta.to()) && b.acceptingStates().contains(tb.to()))
                                    accepting.add(next);
                            }
                        }
                    }
                }
            }

            // A-only actions: only A steps
            for (Transition ta : aTrans) {
                if (!shared.contains(ta.action())) {
                    String next = ta.to() + "|" + sB;
                    transitions.add(new Transition(cur, ta.action(), next));
                    if (visited.add(next)) {
                        states.add(next);
                        queue.add(new String[]{ta.to(), sB});
                        if (a.acceptingStates().contains(ta.to()) && b.acceptingStates().contains(sB))
                            accepting.add(next);
                    }
                }
            }

            // B-only actions: only B steps
            for (Transition tb : bTrans) {
                if (!shared.contains(tb.action())) {
                    String next = sA + "|" + tb.to();
                    transitions.add(new Transition(cur, tb.action(), next));
                    if (visited.add(next)) {
                        states.add(next);
                        queue.add(new String[]{sA, tb.to()});
                        if (a.acceptingStates().contains(sA) && b.acceptingStates().contains(tb.to()))
                            accepting.add(next);
                    }
                }
            }
        }

        Set<String> allActions = new TreeSet<>(aAlpha);
        allActions.addAll(bAlpha);

        return new LTS(
                a.name() + "||" + b.name(),
                initState,
                Collections.unmodifiableList(states),
                Collections.unmodifiableList(new ArrayList<>(allActions)),
                Collections.unmodifiableList(transitions),
                false,
                Collections.unmodifiableSet(accepting),
                LTS.buildIndex(states, initState)
        );
    }

    private static Map<String, List<Transition>> transMap(List<Transition> ts) {
        Map<String, List<Transition>> map = new HashMap<>();
        for (Transition t : ts)
            map.computeIfAbsent(t.from(), k -> new ArrayList<>()).add(t);
        return map;
    }
}
