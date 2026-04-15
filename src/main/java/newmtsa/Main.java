package newmtsa;

import newmtsa.parser.FSPParser;
import newmtsa.parser.ast.*;
import newmtsa.synthesis.SynthesisResult;
import newmtsa.synthesis.gr1.OTFDirectedControledSyntesisGR1;
import newmtsa.synthesis.heuristics.RandomHeuristic;
import newmtsa.synthesis.nonblocking.OTFDirectedControledSyntesisNonBlocking;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import java.io.IOException;
import java.nio.file.Path;


public class Main {

    public static void main(String[] args) throws IOException {
        Path file;
        if(args.length > 0){
            file = Path.of(args[0]);
        }else{
            //file = Path.of(".\\fsp\\Blocking\\Benchmark\\CM\\CM-2-2.fsp");
            //file = Path.of(".\\fsp\\Blocking\\Benchmark\\DP\\DP-2-2.fsp");
            //file = Path.of(".\\fsp\\Blocking\\Benchmark\\TL\\TL-2-2.fsp");
            //file = Path.of(".\\fsp\\Blocking\\Benchmark\\TA\\TA-2-2.fsp");
            //file = Path.of(".\\fsp\\Blocking\\Benchmark\\AT\\AT-2-2.fsp");
            //file = Path.of(".\\fsp\\Blocking\\Benchmark\\BW\\BW-2-2.fsp");
            file = Path.of(".\\fsp\\Blocking\\ControllableFSPs\\GR1Test43.lts");
            //file = Path.of(".\\fsp\\Blocking\\ControllableFSPs\\GR1Test1.lts");
        }

        FSPModel model = FSPParser.parse(file);

        System.out.println("===============================================");
        System.out.println("  File: " + file.getFileName());
        System.out.println("===============================================");

        FSPParser.printModel(model);

        runSynthesis(model, true);
    }

    private static void runSynthesis(FSPModel model, boolean verbose) {
        List<ControllerSpecDef> specs = model.controllerSpecs();
        if (specs.isEmpty()) {
            System.err.println("Must be at least a Goal to use synthesis");
            return;
        }
        ControllerSpecDef spec = specs.get(specs.size() - 1);

        SynthesisResult result;
        if (spec.nonblocking()) {
            System.out.println("\n--- Non-Blocking DCS: " + spec.name() + " ---");
            result = runNonBlocking(model, spec, verbose);
        } else {
            System.out.println("\n--- GR(1) Synthesis: " + spec.name() + " ---");
            result = runGR1(model, spec, verbose);
        }

        System.out.println("  Result: " + (result.isRealizable() ? "REALIZABLE" : "UNREALIZABLE"));
        System.out.println("  States explored     : " + result.statesExplored());
        System.out.println("  Transitions expanded: " + result.transitionsExplored());
    }

    private static SynthesisResult runNonBlocking(FSPModel model, ControllerSpecDef spec, boolean verbose) {
        List<LtlPropertyDef> safetyProps = new ArrayList<>();
        for (String name : spec.safety()) {
            model.ltlProperties().stream()
                    .filter(p -> p.name().equals(name))
                    .findFirst()
                    .ifPresent(safetyProps::add);
        }

        return new OTFDirectedControledSyntesisNonBlocking(
                new ArrayList<>(model.processes()),
                safetyProps,
                new HashSet<>(spec.marking()),
                new HashSet<>(spec.controllable()),
                new RandomHeuristic(22L),
                verbose
        ).run();
    }

    private static SynthesisResult runGR1(FSPModel model, ControllerSpecDef spec, boolean verbose) {
        List<LtlPropertyDef> guarantees = new ArrayList<>();
        for (String name : spec.liveness()) {
            model.asserts().stream()
                    .filter(p -> p.name().equals(name))
                    .findFirst()
                    .or(() -> model.fluents().stream()
                            .filter(f -> f.name().equals(name))
                            .map(f -> new LtlPropertyDef(f.name(), List.of(), f))
                            .findFirst())
                    .ifPresent(guarantees::add);
        }

        List<LtlPropertyDef> assumptions = new ArrayList<>();
        for (String name : spec.assumption()) {
            model.asserts().stream()
                    .filter(p -> p.name().equals(name))
                    .findFirst()
                    .or(() -> model.fluents().stream()
                            .filter(f -> f.name().equals(name))
                            .map(f -> new LtlPropertyDef(f.name(), List.of(), f))
                            .findFirst())
                    .ifPresent(assumptions::add);
        }

        List<LTS> components = new ArrayList<>(model.processes());
        for (LtlPropertyDef g : guarantees) {
            components.add(g.lts());
        }
        for (LtlPropertyDef a : assumptions) {
            if (components.stream().noneMatch(c -> c.name().equals(a.name()))) {
                components.add(a.lts());
            }
        }

        return new OTFDirectedControledSyntesisGR1(
                components, assumptions, guarantees,
                new HashSet<>(spec.controllable()),
                new RandomHeuristic(22L),
                verbose
        ).run();
    }
}
