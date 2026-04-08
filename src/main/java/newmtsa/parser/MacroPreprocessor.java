package newmtsa.parser;

import newmtsa.parser.ast.MacroDef;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Syntactic macro pre-processor for FSP source files.
 *
 * <p>Runs <em>before</em> the ANTLR grammar pass. It:
 * <ol>
 *   <li>Scans the source for {@code def Name(params) = body} declarations.</li>
 *   <li>Removes those declaration lines from the source (they are no longer
 *       needed by the grammar once macros are expanded).</li>
 *   <li>Replaces every call {@code Name(arg, …)} in the remaining source with
 *       the macro body, substituting each formal parameter with the actual
 *       argument.  The substituted body is wrapped in parentheses to keep
 *       operator-precedence intact.</li>
 *   <li>Repeats the expansion pass until no more changes occur (handles macros
 *       that call other macros).</li>
 * </ol>
 */
class MacroPreprocessor {

    /** Result returned to {@link FSPParser}: the expanded source and the macro list. */
    record Result(String source, List<MacroDef> macros) {}

    // ── pattern that matches a single-line def declaration ────────────────────
    // Captures: (1) name  (2) comma-separated params  (3) body text
    private static final Pattern DEF_PATTERN = Pattern.compile(
            "^[ \\t]*def[ \\t]+([A-Za-z_]\\w*)[ \\t]*\\(([^)]*)\\)[ \\t]*=[ \\t]*(.+)$",
            Pattern.MULTILINE);

    static Result process(String source) {
        List<MacroDef> macros = extractMacros(source);
        if (macros.isEmpty()) return new Result(source, macros);

        // Blank out def lines so the grammar does not see them and the
        // substitution loop does not try to expand calls inside their headers.
        String stripped = blankDefLines(source);

        // Expand all macro calls to a fixed point (handles transitive calls).
        String expanded = expandToFixedPoint(stripped, macros);

        return new Result(expanded, macros);
    }

    // ── step 1: parse macro definitions ──────────────────────────────────────

    private static List<MacroDef> extractMacros(String source) {
        List<MacroDef> macros = new ArrayList<>();
        Matcher m = DEF_PATTERN.matcher(source);
        while (m.find()) {
            String       name   = m.group(1);
            List<String> params = parseParams(m.group(2));
            String       body   = m.group(3).trim();
            macros.add(new MacroDef(name, params, body));
        }
        return macros;
    }

    private static List<String> parseParams(String raw) {
        List<String> params = new ArrayList<>();
        for (String p : raw.split(",")) {
            String t = p.trim();
            if (!t.isEmpty()) params.add(t);
        }
        return params;
    }

    // ── step 2: remove def lines ──────────────────────────────────────────────

    private static String blankDefLines(String source) {
        return DEF_PATTERN.matcher(source).replaceAll("");
    }

    // ── step 3: expand macro calls ────────────────────────────────────────────

    private static String expandToFixedPoint(String source, List<MacroDef> macros) {
        for (int pass = 0; pass < 20; pass++) {
            String next = source;
            for (MacroDef macro : macros)
                next = expandMacro(next, macro);
            if (next.equals(source)) return source;
            source = next;
        }
        return source;
    }

    /**
     * Replace every syntactic call {@code Name(arg, …)} with
     * {@code (body[param→arg, …])}.
     */
    private static String expandMacro(String source, MacroDef macro) {
        StringBuilder sb  = new StringBuilder();
        int           pos = 0;

        while (pos < source.length()) {
            int callStart = indexOfCall(source, macro.name(), pos);
            if (callStart == -1) {
                sb.append(source, pos, source.length());
                break;
            }

            // Append everything up to the call site unchanged.
            sb.append(source, pos, callStart);

            // Find the matching closing parenthesis for the argument list.
            int openParen = callStart + macro.name().length();
            int closeParen = matchingClose(source, openParen);
            if (closeParen == -1) {
                // Unbalanced parens — leave this occurrence unexpanded.
                sb.append(source.charAt(callStart));
                pos = callStart + 1;
                continue;
            }

            String argStr = source.substring(openParen + 1, closeParen);
            List<String> args = splitTopLevel(argStr);

            if (args.size() == macro.params().size()) {
                String expanded = substituteParams(macro.body(), macro.params(), args);
                sb.append('(').append(expanded).append(')');
            } else {
                // Arity mismatch — leave unexpanded to avoid silent corruption.
                sb.append(source, callStart, closeParen + 1);
            }
            pos = closeParen + 1;
        }
        return sb.toString();
    }

    /**
     * Find the next position where {@code name(} appears as a complete
     * identifier (not a suffix of a longer word), starting from {@code from}.
     */
    private static int indexOfCall(String source, String name, int from) {
        int i = from;
        while (i <= source.length() - name.length() - 1) {
            int candidate = source.indexOf(name, i);
            if (candidate == -1) return -1;

            boolean prevOk = candidate == 0
                    || !isIdentChar(source.charAt(candidate - 1));
            boolean nextOk = source.charAt(candidate + name.length()) == '(';

            if (prevOk && nextOk) return candidate;
            i = candidate + 1;
        }
        return -1;
    }

    private static boolean isIdentChar(char c) {
        return Character.isLetterOrDigit(c) || c == '_';
    }

    /** Return the index of the {@code )} that closes the {@code (} at {@code openPos}. */
    private static int matchingClose(String source, int openPos) {
        int depth = 0;
        for (int i = openPos; i < source.length(); i++) {
            char c = source.charAt(i);
            if (c == '(') depth++;
            else if (c == ')') { if (--depth == 0) return i; }
        }
        return -1;
    }

    /** Split a comma-separated argument string, respecting nested parentheses. */
    private static List<String> splitTopLevel(String args) {
        List<String> result = new ArrayList<>();
        int depth = 0, start = 0;
        for (int i = 0; i < args.length(); i++) {
            char c = args.charAt(i);
            if      (c == '(') depth++;
            else if (c == ')') depth--;
            else if (c == ',' && depth == 0) {
                result.add(args.substring(start, i).trim());
                start = i + 1;
            }
        }
        result.add(args.substring(start).trim());
        return result;
    }

    /**
     * Replace each formal parameter with its actual argument using whole-word
     * matching, so parameter {@code f} does not replace {@code fork}.
     */
    private static String substituteParams(String body,
                                           List<String> params,
                                           List<String> args) {
        String result = body;
        for (int i = 0; i < params.size(); i++) {
            result = result.replaceAll(
                    "\\b" + Pattern.quote(params.get(i)) + "\\b",
                    Matcher.quoteReplacement(args.get(i)));
        }
        return result;
    }
}
