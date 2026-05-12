package newmtsa.GUI;

import newmtsa.parser.FSPParser;
import newmtsa.parser.ast.ControllerSpecDef;
import newmtsa.parser.ast.FSPModel;
import newmtsa.parser.ast.LTS;
import newmtsa.parser.ast.LtlPropertyDef;
import newmtsa.parser.ast.Transition;

import java.awt.Desktop;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/**
 * Generates an HTML visualization for LTS graphs parsed from an FSP/LTS file.
 */
public final class GUIView {
    private static final Path HTML_DIR = Path.of("src", "main", "java", "newmtsa", "GUI", "HTML");
    private static final Path TEMPLATE_PATH = HTML_DIR.resolve("lts-template.html");
    private static final int MAX_STATE_RENDER = 120;
    static final Path SOL_FILE = Path.of("python", "results", "traces", "DP", "RA_ERG_OPEN", "DP-2-2.sol");
    //static final Path SOL_FILE = Path.of("python", "results", "traces", "DP", "RL", "ROL", "DP-2-2.sol");

    private GUIView() {}

    public static void main(String[] args) throws IOException {
        Path sourceFile;
        if (args.length == 0) {
            sourceFile = Path.of("E:\\Code\\Java\\NewMTSA\\fsp\\Blocking\\Benchmark\\DP\\DP-2-2.fsp").toAbsolutePath().normalize();
        }else{
            sourceFile = Path.of(args[0]).toAbsolutePath().normalize();
        }


        if (!Files.exists(sourceFile)) {
            System.err.println("Input file does not exist: " + sourceFile);
            return;
        }

        FSPModel model = FSPParser.parse(sourceFile);
        Map<String, LTS> ltsMap = collectLts(model);

        if (ltsMap.isEmpty()) {
            System.err.println("No LTS found in parsed model.");
            return;
        }

        Set<String> controllable = new LinkedHashSet<>();
        for (ControllerSpecDef cs : model.controllerSpecs()) {
            controllable.addAll(cs.controllable());
        }
        String traceJson = TraceView.buildTraceJson(SOL_FILE, controllable);

        String html = buildHtmlFromTemplate(ltsMap, sourceFile.getFileName().toString(), traceJson);
        Path output = writeTempHtmlFile(sourceFile, html);

        System.out.println("Temporary HTML generated at: " + output);
        openInBrowser(output);
    }

    private static Map<String, LTS> collectLts(FSPModel model) {
        Map<String, LTS> result = new LinkedHashMap<>();

        for (LTS p : model.processes()) {
            result.put("process:" + p.name(), p);
        }
        for (LTS f : model.fluents()) {
            result.put("fluent:" + f.name(), f);
        }
        for (LtlPropertyDef a : model.asserts()) {
            result.put("assert:" + a.name(), a.lts());
        }
        for (LtlPropertyDef p : model.ltlProperties()) {
            result.put("ltl_property:" + p.name(), p.lts());
        }

        return result;
    }

    private static Path writeTempHtmlFile(Path sourceFile, String html) throws IOException {
        String prefix = stripExtension(sourceFile.getFileName().toString()) + "-lts-viewer-";
        Path output = Files.createTempFile(prefix, ".html");
        Files.writeString(output, html, StandardCharsets.UTF_8);
        output.toFile().deleteOnExit();
        return output;
    }

    private static String buildHtmlFromTemplate(Map<String, LTS> ltsMap, String sourceName, String traceJson) throws IOException {
        if (!Files.exists(TEMPLATE_PATH)) {
            throw new IOException("Template not found: " + TEMPLATE_PATH.toAbsolutePath());
        }

        String template = Files.readString(TEMPLATE_PATH, StandardCharsets.UTF_8);
        StringBuilder graphJson = new StringBuilder();
        StringBuilder selectOptions = new StringBuilder();

        boolean first = true;
        for (Map.Entry<String, LTS> entry : ltsMap.entrySet()) {
            String key = entry.getKey();
            LTS lts = entry.getValue();

            if (!first) {
                graphJson.append(",");
            }
            first = false;

            boolean blocked = lts.states().size() > MAX_STATE_RENDER;
            graphJson.append("\"").append(escapeJson(key)).append("\":{")
                    .append("\"stateCount\":").append(lts.states().size()).append(",")
                    .append("\"blocked\":").append(blocked).append(",")
                    .append("\"dot\":\"").append(escapeJson(lts.toDotString())).append("\",")
                    .append("\"actions\":[");

            for (int a = 0; a < lts.actions().size(); a++) {
                if (a > 0) {
                    graphJson.append(",");
                }
                graphJson.append("\"").append(escapeJson(lts.actions().get(a))).append("\"");
            }
            graphJson.append("]");

            if (blocked) {
                graphJson.append(",\"nodes\":[],\"edges\":[]}");
            } else {
                graphJson.append(",\"nodes\":[");
                for (int i = 0; i < lts.states().size(); i++) {
                    String state = lts.states().get(i);
                    Integer id = lts.stateIndex().getOrDefault(state, i);
                    boolean initial = state.equals(lts.initialState());
                    boolean accepting = lts.acceptingStates().contains(state);

                    if (i > 0) {
                        graphJson.append(",");
                    }

                    String color = initial ? "#b3e5fc" : (accepting ? "#c8e6c9" : "#ffffff");
                    graphJson.append("{")
                            .append("\"id\":").append(id).append(",")
                            .append("\"label\":\"").append(escapeJson(state)).append("\",")
                            .append("\"shape\":\"ellipse\",")
                            .append("\"color\":\"").append(color).append("\"")
                            .append("}");
                }
                graphJson.append("],");

                graphJson.append("\"edges\":[");
                for (int i = 0; i < lts.transitions().size(); i++) {
                    Transition t = lts.transitions().get(i);
                    if (i > 0) {
                        graphJson.append(",");
                    }

                    int fromId = lts.stateIndex().getOrDefault(t.from(), -1);
                    int toId = lts.stateIndex().getOrDefault(t.to(), -1);
                    graphJson.append("{")
                            .append("\"from\":").append(fromId).append(",")
                            .append("\"to\":").append(toId).append(",")
                            .append("\"label\":\"").append(escapeJson(t.action())).append("\",")
                            .append("\"arrows\":\"to\"")
                            .append("}");
                }
                graphJson.append("]}");
            }

            selectOptions.append("<option value=\"")
                    .append(escapeHtml(key))
                    .append("\">")
                    .append(escapeHtml(key))
                    .append("</option>");
        }

        return template
                .replace("{{SOURCE_NAME}}", escapeHtml(sourceName))
                .replace("{{SELECT_OPTIONS}}", selectOptions.toString())
                .replace("{{MAX_STATE_RENDER}}", String.valueOf(MAX_STATE_RENDER))
                .replace("{{GRAPH_DATA}}", "{" + graphJson + "}")
                .replace("{{TRACE_DATA}}", traceJson);
    }

    private static String stripExtension(String name) {
        int idx = name.lastIndexOf('.');
        return (idx > 0) ? name.substring(0, idx) : name;
    }

    private static void openInBrowser(Path htmlFile) {
        try {
            if (Desktop.isDesktopSupported()) {
                Desktop.getDesktop().browse(htmlFile.toUri());
            }
        } catch (Exception ex) {
            System.err.println("Could not open browser automatically: " + ex.getMessage());
        }
    }

    private static String escapeJson(String value) {
        StringBuilder out = new StringBuilder(value.length() + 16);
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\b' -> out.append("\\b");
                case '\f' -> out.append("\\f");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                default -> {
                    if (c < 0x20) out.append(String.format("\\u%04x", (int) c));
                    else out.append(c);
                }
            }
        }
        return out.toString();
    }

    private static String escapeHtml(String value) {
        return value
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\"", "&quot;")
                .replace("'", "&#39;");
    }
}
