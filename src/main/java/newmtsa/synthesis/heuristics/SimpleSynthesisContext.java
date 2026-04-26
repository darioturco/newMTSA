package newmtsa.synthesis.heuristics;

import newmtsa.parser.ast.LTS;
import newmtsa.synthesis.ExtendedTransition;

import java.util.*;

/**
 * Standalone, mutable implementation of {@link SynthesisContext}.
 *
 * <p>Used by {@link newmtsa.synthesis.nonblocking.OTFDirectedControledSyntesisNonBlocking}
 * (replacing the previous anonymous inner class) and in unit tests where the full
 * DCS engine is not needed.
 *
 * <p>{@link #exploredStates} and {@link #goals} are public so callers can mutate
 * them to simulate exploration state without running a synthesis.
 */
public class SimpleSynthesisContext implements SynthesisContext {

    private final List<LTS>         components;
    private final List<Set<String>> componentMarked;
    private final Set<String>       controllable;

    public final Set<String> exploredStates = new LinkedHashSet<>();
    public final Set<String> goals          = new LinkedHashSet<>();

    private final Map<String, List<ExtendedTransition>> successors = new HashMap<>();

    public SimpleSynthesisContext(List<LTS>         components,
                                  List<Set<String>> componentMarked,
                                  Set<String>       controllable) {
        this.components      = List.copyOf(components);
        this.componentMarked = List.copyOf(componentMarked);
        this.controllable    = Set.copyOf(controllable);
    }

    /** Registers {@code state} as explored and records its outgoing transitions. */
    public void addState(String state, List<ExtendedTransition> transitions) {
        exploredStates.add(state);
        successors.put(state, List.copyOf(transitions));
    }

    @Override public List<LTS>         components()      { return components; }
    @Override public List<Set<String>> componentMarked() { return componentMarked; }
    @Override public Set<String>       controllable()    { return controllable; }
    @Override public Set<String>       exploredStates()  { return exploredStates; }
    @Override public Set<String>       goals()           { return goals; }
    @Override public List<ExtendedTransition> successorsOf(String s) {
        return successors.getOrDefault(s, List.of());
    }
}
