package newmtsa.synthesis.gr1;

import newmtsa.synthesis.DCSForPython;
import newmtsa.synthesis.ExtendedTransition;

import java.nio.file.Path;
import java.util.List;

/**
 * Run with: mvn exec:java -Dexec.mainClass=newmtsa.synthesis.gr1.FrontierDataGenerator
 * Prints expected frontier snapshots for DCSBehaviorTest.
 */
public class FrontierDataGenerator {

    static final String[] CONTROLLABLE = {
        "fsp/Blocking/ControllableFSPs/GR1Test3.lts",
        "fsp/Blocking/ControllableFSPs/GR1Test5.lts",
        "fsp/Blocking/ControllableFSPs/GR1Test7.lts",
        "fsp/Blocking/ControllableFSPs/GR1Test10.lts",
        "fsp/Blocking/ControllableFSPs/GR1Test13.lts",
        "fsp/Blocking/ControllableFSPs/GR1Test14.lts",
        "fsp/Blocking/ControllableFSPs/GR1Test15.lts",
        "fsp/Blocking/ControllableFSPs/GR1Test16.lts",
        "fsp/Blocking/ControllableFSPs/GR1Test22.lts",
        "fsp/Blocking/ControllableFSPs/GR1Test24.lts",
    };

    static final String[] NO_CONTROLLABLE = {
        "fsp/Blocking/NoControllableFSPs/GR1Test4.lts",
        "fsp/Blocking/NoControllableFSPs/GR1Test6.lts",
        "fsp/Blocking/NoControllableFSPs/GR1Test11.lts",
        "fsp/Blocking/NoControllableFSPs/GR1Test12.lts",
        "fsp/Blocking/NoControllableFSPs/GR1Test17.lts",
        "fsp/Blocking/NoControllableFSPs/GR1Test18.lts",
        "fsp/Blocking/NoControllableFSPs/GR1Test19.lts",
        "fsp/Blocking/NoControllableFSPs/GR1Test20.lts",
        "fsp/Blocking/NoControllableFSPs/GR1Test21.lts",
        "fsp/Blocking/NoControllableFSPs/GR1Test23.lts",
    };

    static final String TL22 = "fsp/Blocking/Benchmark/TL/TL-2-2.fsp";
    static final int[]  TL22_TRACE = {0, 3, 6, 9, 12, 14, 17, 18, 14, 15, 22, 25, 27, 22, 28, 17};

    public static void main(String[] args) throws Exception {
        System.out.println("// ── CONTROLLABLE ─────────────────────────────────────────────");
        for (String f : CONTROLLABLE) dumpInstance(Path.of(f), null, 12, true);

        System.out.println("\n// ── NO CONTROLLABLE ──────────────────────────────────────────");
        for (String f : NO_CONTROLLABLE) dumpInstance(Path.of(f), null, 12, false);

        System.out.println("\n// ── TL-2-2 ───────────────────────────────────────────────────");
        dumpInstance(Path.of(TL22), TL22_TRACE, TL22_TRACE.length, null);
    }

    static void dumpInstance(Path fsp, int[] trace, int maxExpand, Boolean expectedRealizable) throws Exception {
        String name = fsp.getFileName().toString().replace(".lts", "").replace(".fsp", "");
        DCSForPython dcs = DCSForPython.fromPath(fsp.toString(), "FIRST", "");

        System.out.println("\n        // " + name);
        System.out.println("        // trace=" + (trace == null ? "always-0" : java.util.Arrays.toString(trace)));

        List<List<String>> snapshots = new java.util.ArrayList<>();
        int step = 0;
        while (!dcs.isExplorationEnded() && step < maxExpand) {
            List<ExtendedTransition> frontier = dcs.getFrontier();
            if (frontier.isEmpty()) break;

            // record snapshot
            List<String> snap = new java.util.ArrayList<>();
            for (ExtendedTransition t : frontier) snap.add(t.from() + "|" + t.action() + "|" + t.to());
            snapshots.add(snap);

            int idx = (trace == null) ? 0 : trace[step];
            if (idx >= frontier.size()) {
                System.err.println("  !! trace[" + step + "]=" + idx + " out of range " + frontier.size() + " in " + name);
                idx = 0;
            }
            dcs.expand(idx);
            step++;
        }

        // Print as Java literal
        System.out.println("        List<List<String>> exp_" + name + " = List.of(");
        for (int i = 0; i < snapshots.size(); i++) {
            List<String> snap = snapshots.get(i);
            System.out.print("            List.of(");
            for (int j = 0; j < snap.size(); j++) {
                if (j > 0) System.out.print(", ");
                System.out.print("\"" + snap.get(j) + "\"");
            }
            System.out.print(")");
            if (i < snapshots.size() - 1) System.out.print(",");
            System.out.println();
        }
        System.out.println("        );");

        boolean realized = dcs.isRealizable();
        System.out.println("        // realized=" + realized
            + (expectedRealizable != null ? (realized == expectedRealizable ? " ✓" : " ✗ WRONG!") : ""));
    }
}
