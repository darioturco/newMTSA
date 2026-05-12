package newmtsa.GUI;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Parses a .sol trace file and builds the JSON blob consumed by the Trace Viewer tab.
 * Change SOL_FILE to point at the desired trace before rebuilding.
 */
public final class TraceView {

    private TraceView() {}

    // ── data types ────────────────────────────────────────────────────────────

    record Transition(String from, String action, String to) {}

    record Step(List<Transition> frontier, int selectedIndex, Transition selected) {}

    record ParsedTrace(
        String instanceName, int traceLen, boolean realizable,
        String mode, String heuristic, String heuristicParams,
        List<Step> steps
    ) {}

    // ── public API ────────────────────────────────────────────────────────────

    /** Returns JSON for the trace viewer, or "null" if file is missing or unreadable. */
    public static String buildTraceJson(Path solFile, Set<String> controllable) {
        try {
            if (!Files.exists(solFile)) {
                System.err.println("TraceView: sol file not found: " + solFile.toAbsolutePath());
                return "null";
            }
            return toJson(parseSolFile(solFile), solFile, controllable);
        } catch (Exception e) {
            System.err.println("TraceView: error loading trace: " + e.getMessage());
            return "null";
        }
    }

    // ── parsing ───────────────────────────────────────────────────────────────

    static ParsedTrace parseSolFile(Path path) throws IOException {
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        int i = 0;

        String  instanceName    = lines.get(i++).trim();
        int     traceLen        = Integer.parseInt(lines.get(i++).trim().substring("Trace Len: ".length()));
        boolean realizable      = lines.get(i++).trim().equals("Realizable: True");
        String  mode            = lines.get(i++).trim().substring("Mode: ".length());
        String  heuristic       = lines.get(i++).trim().substring("Heuristic Used: ".length());
        String  heuristicParams = lines.get(i++).trim().substring("Heuristic Parameters: ".length());

        while (i < lines.size() && !lines.get(i).trim().startsWith("[")) i++;

        List<Step> steps = new ArrayList<>();
        while (i < lines.size()) {
            String frontierLine = lines.get(i++).trim();
            if (frontierLine.isEmpty()) continue;
            if (!frontierLine.startsWith("[")) continue;
            if (i >= lines.size()) break;

            String selectedLine = lines.get(i++).trim();
            if (selectedLine.isEmpty()) break;

            List<Transition> frontier = parseFrontier(frontierLine);

            int pipeIdx = selectedLine.indexOf(" | ");
            int selIdx  = Integer.parseInt(selectedLine.substring(0, pipeIdx).trim());
            Transition sel = parseTransition(selectedLine.substring(pipeIdx + 3).trim());

            steps.add(new Step(frontier, selIdx, sel));
        }

        return new ParsedTrace(instanceName, traceLen, realizable, mode,
                               heuristic, heuristicParams, steps);
    }

    static List<Transition> parseFrontier(String line) {
        String inner = line.substring(1, line.length() - 1).trim();
        if (inner.isEmpty()) return List.of();
        // ", " before the next transition's " --[" marker
        String[] parts = inner.split(", (?=.+? --\\[)");
        List<Transition> result = new ArrayList<>(parts.length);
        for (String p : parts) result.add(parseTransition(p.trim()));
        return result;
    }

    static Transition parseTransition(String s) {
        int arrowStart   = s.indexOf(" --[");
        int bracketClose = s.indexOf("]-->", arrowStart);
        String from   = s.substring(0, arrowStart);
        String action = s.substring(arrowStart + 4, bracketClose);
        String to     = s.substring(bracketClose + 5); // past "]--> "
        return new Transition(from, action, to);
    }

    // ── JSON serialisation ────────────────────────────────────────────────────

    static String toJson(ParsedTrace trace, Path solFile, Set<String> controllable) {
        StringBuilder sb = new StringBuilder("{");
        strField(sb, "instanceName",    trace.instanceName());     sb.append(",");
        boolField(sb, "realizable",     trace.realizable());       sb.append(",");
        strField(sb, "mode",            trace.mode());             sb.append(",");
        strField(sb, "heuristic",       trace.heuristic());        sb.append(",");
        strField(sb, "heuristicParams", trace.heuristicParams());  sb.append(",");
        strField(sb, "solFile",         solFile.toString());       sb.append(",");

        sb.append("\"controllable\":[");
        boolean first = true;
        for (String c : controllable) {
            if (!first) sb.append(",");
            first = false;
            sb.append('"').append(esc(c)).append('"');
        }
        sb.append("],\"steps\":[");

        for (int i = 0; i < trace.steps().size(); i++) {
            if (i > 0) sb.append(",");
            Step s = trace.steps().get(i);
            sb.append("{\"frontier\":[");
            for (int j = 0; j < s.frontier().size(); j++) {
                if (j > 0) sb.append(",");
                sb.append(transJson(s.frontier().get(j)));
            }
            sb.append("],\"selectedIndex\":").append(s.selectedIndex())
              .append(",\"selected\":").append(transJson(s.selected()))
              .append("}");
        }
        sb.append("]}");
        return sb.toString();
    }

    private static void strField(StringBuilder sb, String key, String val) {
        sb.append('"').append(key).append("\":\"").append(esc(val)).append('"');
    }

    private static void boolField(StringBuilder sb, String key, boolean val) {
        sb.append('"').append(key).append("\":").append(val);
    }

    private static String transJson(Transition t) {
        return "{\"from\":\"" + esc(t.from()) + "\",\"action\":\"" + esc(t.action())
             + "\",\"to\":\"" + esc(t.to()) + "\"}";
    }

    static String esc(String s) {
        StringBuilder out = new StringBuilder(s.length() + 4);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"'  -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                default   -> out.append(c);
            }
        }
        return out.toString();
    }
}
