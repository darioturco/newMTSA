package newmtsa.parser.ast;

import java.util.List;
import java.util.Set;

/**
 * A Labeled Transition System (LTS) — used for processes, fluents, and asserts.
 *
 * <ul>
 *   <li>{@code name}             — declared name (e.g. {@code Philosopher}, {@code F}, {@code S1})</li>
 *   <li>{@code initialState}     — first reachable state</li>
 *   <li>{@code states}           — every named state</li>
 *   <li>{@code actions}          — unique action labels in the alphabet</li>
 *   <li>{@code transitions}      — one {@link Transition} per action step</li>
 *   <li>{@code isFluent}         — true when created from a {@code fluent} definition</li>
 *   <li>{@code acceptingStates}  — states where the property holds ("on"); empty for plain processes.
 *                                  For fluents: {"on"}.
 *                                  For assert compounds: the set of product-states satisfying the
 *                                  boolean condition (state names use "|" as separator, e.g. "on|off").</li>
 * </ul>
 */
public record LTS(
        String name,
        String initialState,
        List<String> states,
        List<String> actions,
        List<Transition> transitions,
        boolean isFluent,
        Set<String> acceptingStates
) {}
