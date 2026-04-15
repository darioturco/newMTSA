package newmtsa.synthesis;

import java.util.Map;
import java.util.Set;

/**
 * A director (winning strategy) for the controller.
 *
 * <p>For each extended state {@code s}, {@code enabled(s)} returns the set of
 * controllable actions the controller is allowed to take.  Uncontrollable actions
 * are always enabled and are not listed here.
 */
public record Director(Map<String, Set<String>> enabled) {

    /** Returns the controllable actions enabled at extended state {@code state}. */
    public Set<String> at(String state) {
        return enabled.getOrDefault(state, Set.of());
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Director {\n");
        enabled.entrySet().stream()
               .sorted(Map.Entry.comparingByKey())
               .forEach(e -> sb.append("  ").append(e.getKey())
                               .append(" -> ").append(e.getValue()).append("\n"));
        sb.append("}");
        return sb.toString();
    }
}
