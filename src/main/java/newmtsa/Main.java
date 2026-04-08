package newmtsa;

import newmtsa.parser.FSPParser;
import newmtsa.parser.ast.*;

import java.util.List;
import java.util.stream.Collectors;

import java.io.IOException;
import java.nio.file.Path;


public class Main {

    public static void main(String[] args) throws IOException {
        Path file;
        if(args.length > 0){
            file = Path.of(args[0]);
        }else{
            //file = Path.of(".\\fsp\\Benchmark\\CM\\CM-2-2.fsp");
            //file = Path.of(".\\fsp\\Benchmark\\DP\\DP-2-2.fsp");
            //file = Path.of(".\\fsp\\Benchmark\\TL\\TL-2-2.fsp");
            //file = Path.of(".\\fsp\\Benchmark\\TA\\TA-2-2.fsp");
            file = Path.of(".\\fsp\\Benchmark\\AT\\AT-2-2.fsp");
            //file = Path.of(".\\fsp\\Benchmark\\BW\\BW-2-2.fsp");
        }

        FSPModel model = FSPParser.parse(file);

        System.out.println("===============================================");
        System.out.println("  File: " + file.getFileName());
        System.out.println("===============================================");

        printSection("Constants", model.constants(),
                c -> c.name() + " = " + c.value());

        printSection("Ranges", model.ranges(),
                r -> r.name() + " = " + r.init() + ".." + r.end()
                        + "  (size: " + r.size() + ")");

        printSection("Sets", model.sets(),
                s -> s.name() + " = {" + String.join(", ", s.elements()) + "}");

        printSection("Macros", model.macros(),
                m -> m.name() + "(" + String.join(", ", m.params()) + ")"
                        + " = " + m.body());

        printSection("Actions", List.of(
                model.actions().stream().sorted().collect(Collectors.joining(", ", "{ ", " }"))
        ), a -> a);

        printProcesses(model.processes()); // LTS

        printSection("Composites (ParallelCompositionLazy)", model.composites(),
                c -> c.name() + "  components: ["
                        + c.components().stream().map(LTS::name)
                               .reduce((a, b) -> a + ", " + b).orElse("unresolved")
                        + "]");

        printSection("Fluents", model.fluents(),
                f -> f.name()
                    + " = <{" + String.join(", ", f.initiatingActions()) + "}"
                    + ", {" + String.join(", ", f.terminatingActions()) + "}>");

        printSection("Asserts", model.asserts(),
                a -> a.name());

        printSection("LTL Properties", model.ltlProperties(),
                p -> p.name());

        printSection("Controller Specs", model.controllerSpecs(), cs -> {
            StringBuilder sb = new StringBuilder(cs.name()).append(" {\n");
            if (!cs.liveness().isEmpty())
                sb.append("    liveness     = ").append(cs.liveness()).append("\n");
            if (!cs.safety().isEmpty())
                sb.append("    safety       = ").append(cs.safety()).append("\n");
            if (!cs.assumption().isEmpty())
                sb.append("    assumption   = ").append(cs.assumption()).append("\n");
            if (!cs.controllable().isEmpty())
                sb.append("    controllable = ").append(cs.controllable()).append("\n");
            if (!cs.marking().isEmpty())
                sb.append("    marking      = ").append(cs.marking()).append("\n");
            if (cs.nonblocking())
                sb.append("    nonblocking\n");
            sb.append("  }");
            return sb.toString();
        });

        if (!model.errors().isEmpty()) {
            System.out.println("\n--- Parse Errors (" + model.errors().size() + ") ---");
            model.errors().forEach(e -> System.out.println("  " + e));
        }
    }

    private static void printProcesses(List<LTS> processes) {
        System.out.println("\n--- Processes / LTS (" + processes.size() + ") ---");
        for (LTS p : processes) {
            System.out.println("\n  " + p.name() + "  (initial: " + p.initialState() + ")");
            System.out.println("    States (" + p.states().size() + "): "
                    + String.join(", ", p.states()));
            System.out.println("    Actions (" + p.actions().size() + "): "
                    + String.join(", ", p.actions()));
            System.out.println("    Transitions (" + p.transitions().size() + "):");
            for (Transition t : p.transitions()) {
                System.out.println("      " + t.from() + " --[" + t.action() + "]--> " + t.to());
            }
        }
    }

    private static <T> void printSection(String title, List<T> items, Formatter<T> fmt) {
        System.out.println("\n--- " + title + " (" + items.size() + ") ---");
        if (items.isEmpty()) {
            System.out.println("  (none)");
        } else {
            items.forEach(item -> System.out.println("  " + fmt.format(item)));
        }
    }

    @FunctionalInterface
    private interface Formatter<T> {
        String format(T item);
    }
}
