package newmtsa.synthesis.gr1;

import newmtsa.parser.FSPParser;
import newmtsa.parser.ast.ControllerSpecDef;
import newmtsa.parser.ast.FSPModel;
import newmtsa.parser.ast.LTS;
import newmtsa.parser.ast.LtlPropertyDef;
import newmtsa.synthesis.Director;
import newmtsa.synthesis.heuristics.RandomHeuristic;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

public class TestGR1Debug {
    public static void main(String[] args) throws IOException {
        for (String f : new String[]{
            "fsp/Blocking/ControllableFSPs/GR1Test43.lts",
            "fsp/Blocking/ControllableFSPs/GR1Test44.lts",
            "fsp/Blocking/ControllableFSPs/GR1Test45.lts"
        }) {
            Path file = Paths.get(f);
            System.out.println("\n══════════════════════════════════════════════════════");
            System.out.println("FILE: " + file.getFileName());
            System.out.println("══════════════════════════════════════════════════════");

            FSPModel model = FSPParser.parse(file);
            ControllerSpecDef spec = model.controllerSpecs().stream()
                .filter(s -> !s.liveness().isEmpty()).findFirst().get();

            List<LtlPropertyDef> guarantees = new ArrayList<>();
            for (String name : spec.liveness()) {
                model.asserts().stream()
                    .filter(p -> p.name().equals(name)).findFirst()
                    .or(() -> model.fluents().stream()
                        .filter(fl -> fl.name().equals(name))
                        .map(fl -> new LtlPropertyDef(fl.name(), List.of(), fl))
                        .findFirst())
                    .ifPresent(guarantees::add);
            }

            List<LtlPropertyDef> assumptions = new ArrayList<>();
            for (String name : spec.assumption()) {
                model.asserts().stream()
                    .filter(p -> p.name().equals(name)).findFirst()
                    .or(() -> model.fluents().stream()
                        .filter(fl -> fl.name().equals(name))
                        .map(fl -> new LtlPropertyDef(fl.name(), List.of(), fl))
                        .findFirst())
                    .ifPresent(assumptions::add);
            }

            List<LTS> components = new ArrayList<>(model.processes());
            for (LtlPropertyDef g : guarantees) components.add(g.lts());
            for (LtlPropertyDef a : assumptions) {
                if (components.stream().noneMatch(c -> c.name().equals(a.name())))
                    components.add(a.lts());
            }

            System.out.println("assumptions=" + spec.assumption()
                + "  guarantees=" + spec.liveness()
                + "  controllable=" + spec.controllable());

            Director result = new OTFDirectedControledSyntesisGR1(
                components, assumptions, guarantees,
                new HashSet<>(spec.controllable()),
                new RandomHeuristic(42L),
                true   // verbose
            ).run();

            System.out.println("RESULT: " + (result.isRealizable() ? "REALIZABLE" : "UNREALIZABLE")
                + " (states=" + result.statesExplored()
                + ", transitions=" + result.transitionsExplored() + ")");
        }
    }
}
