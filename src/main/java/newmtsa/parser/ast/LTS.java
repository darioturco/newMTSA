package newmtsa.parser.ast;

import newmtsa.parser.FSPParser;

import java.io.IOException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.stream.Stream;

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
 *   <li>{@code stateIndex}       — maps each state name to its unique integer ID (0-based).
 *                                  The initial state is always assigned ID 0.</li>
 * </ul>
 */
public record LTS(
        String name,
        String initialState,
        List<String> states,
        List<String> actions,
        List<Transition> transitions,
        boolean isFluent,
        Set<String> acceptingStates,
        Map<String, Integer> stateIndex
) {
    private record NamedLTS(String tag, LTS lts) {}

    /**
     * Builds a state-to-ID map from a states list.
     * The initial state is always assigned ID 0; remaining states receive IDs 1..n-1
     * in their original list order.
     */
    public static Map<String, Integer> buildIndex(List<String> states, String initialState) {
        Map<String, Integer> idx = new LinkedHashMap<>();
        idx.put(initialState, 0);
        for (String s : states) {
            if (!s.equals(initialState)) idx.put(s, idx.size());
        }
        return Collections.unmodifiableMap(idx);
    }

    /**
     * Returns this LTS as a Graphviz DOT directed graph.
     */
    public String toDotString() {
        StringBuilder dot = new StringBuilder();
        dot.append("digraph \"").append(escapeDot(name)).append("\" {\n");
        dot.append("  rankdir=LR;\n");
        dot.append("  node [shape=circle];\n");
        dot.append("  \"__start\" [shape=point, label=\"\"];\n");
        dot.append("  \"__start\" -> \"").append(escapeDot(initialState)).append("\";\n");

        for (String state : states) {
            if (acceptingStates.contains(state)) {
                dot.append("  \"").append(escapeDot(state))
                        .append("\" [shape=doublecircle];\n");
            } else {
                dot.append("  \"").append(escapeDot(state))
                        .append("\" [shape=circle];\n");
            }
        }

        for (Transition transition : transitions) {
            dot.append("  \"").append(escapeDot(transition.from())).append("\" -> \"")
                    .append(escapeDot(transition.to())).append("\" [label=\"")
                    .append(escapeDot(transition.action())).append("\"];\n");
        }
        dot.append("}\n");
        return dot.toString();
    }

    /**
     * Recursively scans {@code folder} for .lts files, parses each file, and writes
     * one .dot file per parsed LTS in the same directory as the source file.
     */
    public static void convertLtsFilesInFolderToDot(Path folder) throws IOException {
        if (!Files.isDirectory(folder)) {
            throw new IOException("Not a directory: " + folder);
        }

        try (Stream<Path> paths = Files.walk(folder)) {
            for (Path source : paths.filter(Files::isRegularFile)
                    .filter(LTS::isLtsFile)
                    .toList()) {
                convertSingleFileToDot(source);
            }
        }
    }

    private static void convertSingleFileToDot(Path sourceFile) throws IOException {
        FSPModel model = FSPParser.parse(sourceFile);
        List<NamedLTS> all = collectAllLts(model);
        if (all.isEmpty()) return;

        String baseName = stripExtension(sourceFile.getFileName().toString());
        Path parent = sourceFile.getParent();
        Map<String, Integer> usedNames = new LinkedHashMap<>();

        for (NamedLTS item : all) {
            String rawName = baseName + "-" + item.tag() + "-" + item.lts().name();
            String safeName = sanitizeFileName(rawName);
            int index = usedNames.getOrDefault(safeName, 0);
            usedNames.put(safeName, index + 1);

            String finalName = index == 0 ? safeName : safeName + "-" + index;
            Path out = parent.resolve(finalName + ".dot");
            Files.writeString(out, item.lts().toDotString(), StandardCharsets.UTF_8);
        }
    }

    private static List<NamedLTS> collectAllLts(FSPModel model) {
        List<NamedLTS> result = new ArrayList<>();
        for (LTS p : model.processes()) {
            result.add(new NamedLTS("process", p));
        }
        for (LTS f : model.fluents()) {
            result.add(new NamedLTS("fluent", f));
        }
        for (LtlPropertyDef a : model.asserts()) {
            result.add(new NamedLTS("assert", a.lts()));
        }
        for (LtlPropertyDef p : model.ltlProperties()) {
            result.add(new NamedLTS("ltl_property", p.lts()));
        }
        return result;
    }

    private static boolean isLtsFile(Path path) {
        String name = path.getFileName().toString().toLowerCase();
        return name.endsWith(".lts");
    }

    private static String stripExtension(String name) {
        int idx = name.lastIndexOf('.');
        return (idx > 0) ? name.substring(0, idx) : name;
    }

    private static String sanitizeFileName(String name) {
        return name.replaceAll("[\\\\/:*?\"<>|]", "_");
    }

    private static String escapeDot(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    public static void main(String[] args) throws IOException {
        String folder_path = "C:\\Users\\diort\\Downloads\\doc\\data\\krka_et_al_FSE14\\reference_models\\";
        LTS.convertLtsFilesInFolderToDot(Path.of(folder_path));
    }
}
