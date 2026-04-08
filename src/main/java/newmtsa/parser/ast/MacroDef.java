package newmtsa.parser.ast;

import java.util.List;

/**
 * A macro definition: {@code def LeftF(f) = f == 0 ? N-1 : f-1}
 *
 * <p>Macros are purely syntactic: every call site {@code LeftF(expr)} is replaced
 * with the body expression with the parameter substituted, before the ANTLR grammar
 * sees the source.
 *
 * <ul>
 *   <li>{@code name}   — macro name (e.g. {@code LeftF})</li>
 *   <li>{@code params} — formal parameter names (e.g. {@code ["f"]})</li>
 *   <li>{@code body}   — replacement template (e.g. {@code "f == 0 ? N-1 : f-1"})</li>
 * </ul>
 */
public record MacroDef(String name, List<String> params, String body) {}
