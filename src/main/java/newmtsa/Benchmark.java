package newmtsa;

import newmtsa.parser.FSPParser;
import newmtsa.parser.ast.ControllerSpecDef;
import newmtsa.parser.ast.FSPModel;
import newmtsa.parser.ast.LTS;
import newmtsa.parser.ast.LtlPropertyDef;
import newmtsa.synthesis.SynthesisResult;
import newmtsa.synthesis.heuristics.HeuristicType;
import newmtsa.synthesis.nonblocking.OTFDirectedControledSyntesisNonBlocking;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Stream;

/**
 * Benchmark runner that evaluates all non-blocking heuristics across all
 * instance families (AT, BW, CM, DP, TA, TL).
 *
 * <p>An instance is considered "solved" if synthesis finishes in fewer than
 * {@link #EXPANSION_LIMIT} transitions expanded.  Results are written to
 * {@code benchmark_results.csv} in the project root.
 *
 * <p>Run with: {@code mvn exec:java -Dexec.mainClass=newmtsa.Benchmark}
 */
public class Benchmark {

    static final int EXPANSION_LIMIT = 2500;

    static final String BENCHMARK_DIR = "fsp/NonBlocking/Benchmark";

    static final String[] FAMILIES = {"AT", "BW", "CM", "DP", "TA", "TL"};

    static final HeuristicType[] HEURISTICS = {
            HeuristicType.RANDOM,
            HeuristicType.FIRST,
            HeuristicType.BFS,
            HeuristicType.RA,
            HeuristicType.RA_ER,
    };

    public static void main(String[] args) throws IOException {
        Path experimentsDir = Paths.get("Experiments");
        Files.createDirectories(experimentsDir);
        String csvPath = experimentsDir.resolve("benchmark_results.csv").toString();

        System.out.println("=== NonBlocking Heuristic Benchmark ===");
        System.out.printf("Expansion limit : %d%n", EXPANSION_LIMIT);
        System.out.printf("Output CSV      : %s%n%n", csvPath);

        try (PrintWriter csv = new PrintWriter(new FileWriter(csvPath))) {
            // Header
            csv.print("family,n,k,heuristic,solved,transitions");
            csv.println();

            for (String family : FAMILIES) {
                System.out.printf("--- Family: %s ---%n", family);
                Path dir = Paths.get(BENCHMARK_DIR, family);

                // Collect all files for this family, sorted by (n, k)
                List<Path> files;
                try (Stream<Path> stream = Files.walk(dir)) {
                    files = stream
                            .filter(p -> p.toString().endsWith(".fsp") || p.toString().endsWith(".lts"))
                            .sorted(Comparator.comparing(p -> parseNK(p.getFileName().toString())))
                            .toList();
                }

                // Per-heuristic: the lowest N at which it first failed.
                // Any instance with N > this threshold is skipped (assumed to fail).
                Map<HeuristicType, Integer> failedFromN = new EnumMap<>(HeuristicType.class);

                for (Path file : files) {
                    int[] nk = parseNKArray(file.getFileName().toString());
                    int n = nk[0], k = nk[1];

                    FSPModel model = null;

                    // Collect results for the end-of-instance summary line
                    Map<HeuristicType, Integer> instanceResults = new EnumMap<>(HeuristicType.class);

                    for (HeuristicType type : HEURISTICS) {
                        // Skip if this heuristic already failed at a smaller N
                        if (n > failedFromN.getOrDefault(type, Integer.MAX_VALUE)) {
                            csv.printf("%s,%d,%d,%s,%b,%d%n",
                                    family, n, k, type.name(), false, -1);
                            instanceResults.put(type, -1);
                            continue;
                        }

                        // Lazy-parse the model only when at least one heuristic needs it
                        if (model == null) {
                            try {
                                model = FSPParser.parse(file);
                            } catch (Exception e) {
                                System.err.printf("  PARSE ERROR %s: %s%n", file.getFileName(), e.getMessage());
                                break; // skip all heuristics for this file
                            }
                            ControllerSpecDef spec = findNonBlockingSpec(model);
                            if (spec == null) {
                                System.err.printf("  NO SPEC in %s — skipping%n", file.getFileName());
                                model = null;
                                break;
                            }
                        }

                        ControllerSpecDef spec = findNonBlockingSpec(model);
                        List<LtlPropertyDef> safetyProps = collectSafetyProps(model, spec);
                        List<LTS> components = new ArrayList<>(model.processes());

                        BenchmarkRun run = runWithLimit(components, safetyProps, spec, type);
                        boolean solved = run.transitions < EXPANSION_LIMIT;

                        if (!solved) {
                            failedFromN.merge(type, n, Math::min);
                        }

                        csv.printf("%s,%d,%d,%s,%b,%d%n",
                                family, n, k, type.name(), solved, run.transitions);
                        instanceResults.put(type, run.transitions);
                    }

                    // Print one summary line per instance
                    StringBuilder sb = new StringBuilder();
                    sb.append(String.format("  N=%2d K=%2d |", n, k));
                    for (HeuristicType type : HEURISTICS) {
                        int t = instanceResults.getOrDefault(type, -1);
                        String val = (t == -1) ? "  skip" : String.format("%6d", t);
                        sb.append(String.format(" %s:%s", type.name(), val));
                    }
                    System.out.println(sb);
                }
            }
        }

        System.out.printf("%nBenchmark complete. Results written to: %s%n", csvPath);
    }

    // ── synthesis ─────────────────────────────────────────────────────────────

    private static BenchmarkRun runWithLimit(List<LTS> components,
                                             List<LtlPropertyDef> safetyProps,
                                             ControllerSpecDef spec,
                                             HeuristicType type) {
        try {
            SynthesisResult result = new OTFDirectedControledSyntesisNonBlocking(
                    components,
                    safetyProps,
                    new HashSet<>(spec.marking()),
                    new HashSet<>(spec.controllable()),
                    type.create(),
                    false,
                    EXPANSION_LIMIT
            ).run();
            return new BenchmarkRun(result.transitionsExplored());
        } catch (Exception e) {
            // Treat exceptions (e.g. stack overflow on huge instances) as unsolved
            return new BenchmarkRun(Integer.MAX_VALUE);
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    private static ControllerSpecDef findNonBlockingSpec(FSPModel model) {
        List<ControllerSpecDef> specs = model.controllerSpecs();
        for (int i = specs.size() - 1; i >= 0; i--) {
            if (specs.get(i).nonblocking()) return specs.get(i);
        }
        return null;
    }

    private static List<LtlPropertyDef> collectSafetyProps(FSPModel model, ControllerSpecDef spec) {
        List<LtlPropertyDef> safety = new ArrayList<>();
        for (String name : spec.safety()) {
            model.ltlProperties().stream()
                    .filter(p -> p.name().equals(name))
                    .findFirst()
                    .ifPresent(safety::add);
        }
        return safety;
    }

    /**
     * Parses the N-K pair from a filename like {@code AT-3-7.fsp}.
     * Returns a {@code long} that sorts by (n, k) numerically.
     */
    private static long parseNK(String filename) {
        int[] nk = parseNKArray(filename);
        return ((long) nk[0] << 32) | (nk[1] & 0xFFFFFFFFL);
    }

    private static int[] parseNKArray(String filename) {
        // e.g. "AT-3-7.fsp" → [3, 7]
        String base = filename.replaceAll("\\.[^.]+$", ""); // strip extension
        String[] parts = base.split("-");
        if (parts.length < 3) return new int[]{0, 0};
        try {
            return new int[]{Integer.parseInt(parts[1]), Integer.parseInt(parts[2])};
        } catch (NumberFormatException e) {
            return new int[]{0, 0};
        }
    }

    private record BenchmarkRun(int transitions) {}
}
