package newmtsa.parser.ast;

import java.util.List;

/**
 * A Labeled Transition System (LTS) parsed from an FSP process definition.
 *
 * <ul>
 *   <li>{@code name}         — declared process name (e.g. {@code Philosopher})</li>
 *   <li>{@code initialState} — first reachable state; equals {@code name} when the
 *                              body is an inline choice, otherwise the aliased state</li>
 *   <li>{@code states}       — every named state, including auto-generated intermediate
 *                              states from action chains (prefixed with {@code _})</li>
 *   <li>{@code actions}      — unique base action labels (dot-notation kept, indices stripped)</li>
 *   <li>{@code transitions}  — one {@link Transition} per action step</li>
 * </ul>
 */
public record LTS(
        String name,
        String initialState,
        List<String> states,
        List<String> actions,
        List<Transition> transitions
) {}
