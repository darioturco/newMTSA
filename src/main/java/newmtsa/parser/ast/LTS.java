package newmtsa.parser.ast;

import java.util.List;

/**
 * A Labeled Transition System (LTS) parsed from an FSP process or fluent definition.
 *
 * <ul>
 *   <li>{@code name}         — declared name (e.g. {@code Philosopher} or {@code F})</li>
 *   <li>{@code initialState} — first reachable state</li>
 *   <li>{@code states}       — every named state, including auto-generated intermediate
 *                              states from action chains (prefixed with {@code _})</li>
 *   <li>{@code actions}      — unique action labels</li>
 *   <li>{@code transitions}  — one {@link Transition} per action step</li>
 *   <li>{@code isFluent}     — true when this LTS was created from a {@code fluent} definition;
 *                              its two states are {@code "off"} (initial) and {@code "on"}</li>
 * </ul>
 */
public record LTS(
        String name,
        String initialState,
        List<String> states,
        List<String> actions,
        List<Transition> transitions,
        boolean isFluent
) {}
