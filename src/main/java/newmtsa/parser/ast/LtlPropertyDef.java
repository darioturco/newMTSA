package newmtsa.parser.ast;

import java.util.List;

/**
 * An LTL/FLTL property definition (assert or ltlProperty).
 *
 * <ul>
 *   <li>{@code name}    — declared property name</li>
 *   <li>{@code fluents} — names of the fluents composed to form this property</li>
 *   <li>{@code lts}     — the LTS resulting from the composition of those fluents</li>
 * </ul>
 */
public record LtlPropertyDef(String name, List<String> fluents, LTS lts) {}
