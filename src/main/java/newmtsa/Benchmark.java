package newmtsa;

import newmtsa.parser.FSPParser;
import newmtsa.parser.ast.ControllerSpecDef;
import newmtsa.parser.ast.FSPModel;
import newmtsa.parser.ast.LTS;
import newmtsa.parser.ast.LtlPropertyDef;
import newmtsa.synthesis.Director;
import newmtsa.synthesis.gr1.OTFDirectedControledSyntesisGR1;
import newmtsa.synthesis.heuristics.SuperDFSHeuristic;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Stream;

/**
 * Benchmark runner: SuperDFS heuristic on blocking (GR1) instances.
 *
 * <p>Families: AT, BW, CM, DP, TL (TA excluded — picker not yet implemented).
 * Budget: 15 000 expansions per instance.
 * Output CSV: same format as RA.csv → Instance,N,K,Name,Transitions,Time
 *
 * <p>Run: {@code mvn exec:java -Dexec.mainClass=newmtsa.Benchmark}
 */
public class Benchmark {

    static final int    EXPANSION_LIMIT   = 15_000;
    static final String BENCHMARK_DIR     = "fsp/Blocking/Benchmark";
    static final String[] FAMILIES        = {"TA", "AT", "BW", "CM", "DP", "TL"};
    static final boolean  TIMEOUT_CUTS    = true;  // false = run every instance regardless of prior timeouts

    public static void main(String[] args) throws IOException {
        Path outDir = Paths.get("Experiments");
        Files.createDirectories(outDir);
        String csvPath = outDir.resolve("SuperDFS_results.csv").toString();

        System.out.println("=== SuperDFS Blocking Benchmark ===");
        System.out.printf("Budget: %d  |  Output: %s%n%n", EXPANSION_LIMIT, csvPath);

        try (PrintWriter csv = new PrintWriter(new FileWriter(csvPath, true))) {

            for (String family : FAMILIES) {
                System.out.printf("--- Family: %s ---%n", family);
                Path dir = Paths.get(BENCHMARK_DIR, family);

                List<Path> files;
                try (Stream<Path> stream = Files.walk(dir)) {
                    files = stream
                            .filter(p -> p.toString().endsWith(".fsp") || p.toString().endsWith(".lts"))
                            .sorted(Comparator.comparing(p -> parseNK(p.getFileName().toString())))
                            .toList();
                }

                int skipFromN = Integer.MAX_VALUE; // skip all n >= this
                int currentN  = -1;
                int skipFromK = Integer.MAX_VALUE; // skip k >= this within currentN

                for (Path file : files) {
                    int[] nk = parseNKArray(file.getFileName().toString());
                    int n = nk[0], k = nk[1];

                    if (n != currentN) { currentN = n; skipFromK = Integer.MAX_VALUE; }

                    if (TIMEOUT_CUTS && (n >= skipFromN || k >= skipFromK)) {
                        csv.printf("%s,%d,%d,,%d,%d%n", family, n, k, EXPANSION_LIMIT, -1);
                        System.out.printf("  N=%2d K=%2d | SKIPPED%n", n, k);
                        continue;
                    }

                    BenchmarkRun run = runInstance(file, family, n, k);
                    boolean solved = run.transitions < EXPANSION_LIMIT;
                    long timeOut = solved ? run.timeMs : -1;

                    csv.printf("%s,%d,%d,,%d,%d%n", family, n, k, run.transitions, timeOut);
                    System.out.printf("  N=%2d K=%2d | %s  transitions=%d%n",
                            n, k, solved ? "SOLVED " : "TIMEOUT", run.transitions);

                    if (!solved && TIMEOUT_CUTS) {
                        skipFromK = k + 1;          // skip k > j for same n
                        if (k == 1) skipFromN = n;  // k=1 failed → skip all n >= i
                    }
                }
            }
        }

        System.out.printf("%nDone. CSV: %s%n", csvPath);
    }

    // ── synthesis ─────────────────────────────────────────────────────────────

    private static BenchmarkRun runInstance(Path file, String family, int n, int k) {
        FSPModel model;
        try {
            model = FSPParser.parse(file);
        } catch (Exception e) {
            System.err.printf("  PARSE ERROR %s: %s%n", file.getFileName(), e.getMessage());
            return new BenchmarkRun(EXPANSION_LIMIT, -1);
        }

        ControllerSpecDef spec = findLivenessSpec(model);
        if (spec == null) {
            System.err.printf("  NO LIVENESS SPEC in %s%n", file.getFileName());
            return new BenchmarkRun(EXPANSION_LIMIT, -1);
        }

        List<LtlPropertyDef> guarantees = collectProps(model, spec.liveness());
        List<LtlPropertyDef> assumptions = collectProps(model, spec.assumption());

        List<LTS> components = new ArrayList<>(model.processes());
        for (LtlPropertyDef g : guarantees) components.add(g.lts());
        for (LtlPropertyDef a : assumptions) {
            if (components.stream().noneMatch(c -> c.name().equals(a.name()))) components.add(a.lts());
        }

        SuperDFSHeuristic h = new SuperDFSHeuristic(family, n, k);

        try {
            long start = System.nanoTime();
            Director result = new OTFDirectedControledSyntesisGR1(
                    components, assumptions, guarantees,
                    new HashSet<>(spec.controllable()), h, false)
                    .run(EXPANSION_LIMIT);
            long elapsedMs = (System.nanoTime() - start) / 1_000_000;
            int t = result.transitionsExplored();
            if (t > EXPANSION_LIMIT) t = EXPANSION_LIMIT;
            return new BenchmarkRun(t, elapsedMs);
        } catch (Exception e) {
            return new BenchmarkRun(EXPANSION_LIMIT, -1);
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    private static ControllerSpecDef findLivenessSpec(FSPModel model) {
        List<ControllerSpecDef> specs = model.controllerSpecs();
        for (int i = specs.size() - 1; i >= 0; i--) {
            if (!specs.get(i).liveness().isEmpty()) return specs.get(i);
        }
        return null;
    }

    private static List<LtlPropertyDef> collectProps(FSPModel model, List<String> names) {
        List<LtlPropertyDef> result = new ArrayList<>();
        for (String name : names) {
            model.asserts().stream().filter(p -> p.name().equals(name)).findFirst()
                .or(() -> model.fluents().stream()
                    .filter(f -> f.name().equals(name))
                    .map(f -> new LtlPropertyDef(f.name(), List.of(), f))
                    .findFirst())
                .ifPresent(result::add);
        }
        return result;
    }

    private static long parseNK(String filename) {
        int[] nk = parseNKArray(filename);
        return ((long) nk[0] << 32) | (nk[1] & 0xFFFFFFFFL);
    }

    private static int[] parseNKArray(String filename) {
        String base = filename.replaceAll("\\.[^.]+$", "");
        String[] parts = base.split("-");
        if (parts.length < 3) return new int[]{0, 0};
        try {
            return new int[]{Integer.parseInt(parts[1]), Integer.parseInt(parts[2])};
        } catch (NumberFormatException e) {
            return new int[]{0, 0};
        }
    }

    private record BenchmarkRun(int transitions, long timeMs) {}
}
