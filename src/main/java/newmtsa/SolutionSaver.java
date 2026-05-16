package newmtsa;

import newmtsa.synthesis.Director;
import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.heuristics.HeuristicType;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public final class SolutionSaver {

    private SolutionSaver() {}

    public static void saveSol(Path file, Director result, String mode, List<Main.TraceStep> trace, HeuristicType heuristic, String featureType) throws IOException {
        String fileName = file.getFileName().toString();
        String instanceName = fileName.contains(".")
                ? fileName.substring(0, fileName.lastIndexOf('.'))
                : fileName;
        String family = (file.getParent() != null) ? file.getParent().getFileName().toString() : "unknown";

        Path outDir;
        if (heuristic == HeuristicType.RL || heuristic == HeuristicType.MCTS_RL) {
            outDir = Path.of("python", "results", "traces", family, heuristic.name(), featureType);
        } else {
            outDir = Path.of("python", "results", "traces", family, heuristic.name());
        }
        Files.createDirectories(outDir);
        Path outFile = outDir.resolve(instanceName + ".sol");

        try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(outFile))) {
            pw.println(instanceName);
            pw.println("Trace Len: " + trace.size());
            pw.println("Realizable: " + (result.isRealizable() ? "True" : "False"));
            pw.println("Mode: " + mode);
            pw.println("Heuristic Used: " + heuristic.name());
            pw.println("Heuristic Parameters: " + heuristic.params());
            pw.println("Trace: ");
            pw.println();
            for (Main.TraceStep step : trace) {
                pw.println(formatFrontier(step.frontier()));
                pw.println(step.index() + " | " + step.selected().format());
            }
            if (result.isRealizable()) {
                pw.println();
                pw.println("Director Transitions:");
                pw.println();
                for (var entry : result.enabled().entrySet()) {
                    for (ExtendedTransition t : entry.getValue()) {
                        pw.println("  " + t.format());
                    }
                }
            }
        }

        System.out.println("  Saved: " + outFile);
    }

    private static String formatFrontier(List<ExtendedTransition> frontier) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < frontier.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(frontier.get(i).format());
        }
        sb.append("]");
        return sb.toString();
    }
}
