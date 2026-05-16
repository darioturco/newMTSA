package newmtsa.synthesis;

import java.util.List;
import java.util.Map;

/**
 * A director (winning strategy) for the controller.
 *
 * <p>For each extended state {@code s}, {@code enabled(s)} returns the list of
 * transitions the controller is allowed to take.  Uncontrollable actions
 * are always enabled and are not listed here.
 */
public final class Director {

    public enum TerminationReason { GOAL, ERROR, NONE, FRONTIER_EMPTY, BUDGET_EXHAUSTED }

    private final boolean realizable;
    private final Map<String, List<ExtendedTransition>> enabled;
    private final int statesExplored;
    private final int transitionsExplored;
    private final TerminationReason terminationReason;

    private Director(boolean realizable, Map<String, List<ExtendedTransition>> enabled, int statesExplored, int transitionsExplored, TerminationReason terminationReason) {
        this.realizable = realizable;
        this.enabled = enabled;
        this.statesExplored = statesExplored;
        this.transitionsExplored = transitionsExplored;
        this.terminationReason = terminationReason;
    }

    public static Director realizable(Map<String, List<ExtendedTransition>> enabled, int statesExplored, int transitionsExplored) {
        return new Director(true, enabled, statesExplored, transitionsExplored, TerminationReason.GOAL);
    }

    public static Director unrealizable(int statesExplored, int transitionsExplored, TerminationReason reason) {
        return new Director(false, Map.of(), statesExplored, transitionsExplored, reason);
    }

    public boolean isRealizable()                              { return realizable; }
    public Map<String, List<ExtendedTransition>> enabled()     { return enabled; }
    public int statesExplored()                                { return statesExplored; }
    public int transitionsExplored()                           { return transitionsExplored; }
    public TerminationReason terminationReason()               { return terminationReason; }

    /** Returns the transitions enabled at extended state {@code state}. */
    public List<ExtendedTransition> at(String state) {
        return enabled.getOrDefault(state, List.of());
    }

    @Override
    public String toString() {
        if (!realizable) return "Director (UNREALIZABLE)";
        StringBuilder sb = new StringBuilder("Director {\n");
        enabled.entrySet().stream()
               .sorted(Map.Entry.comparingByKey())
               .forEach(e -> {
                   sb.append("  ").append(e.getKey()).append(":\n");
                   for (ExtendedTransition t : e.getValue()) {
                       sb.append("    ").append(t.from()).append(" --[")
                         .append(t.action()).append("]--> ").append(t.to()).append("\n");
                   }
               });
        sb.append("}");
        return sb.toString();
    }
}
