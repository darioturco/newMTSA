package newmtsa.synthesis.nonblocking;

import newmtsa.parser.FSPParser;
import newmtsa.parser.ast.*;
import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.heuristics.Heuristic;
import org.junit.jupiter.api.Test;
import java.nio.file.Path;
import java.util.*;

class DebugTL22 {
    @Test void debugFrontier() throws Exception {
        FSPModel model = FSPParser.parse(Path.of("fsp/NonBlocking/Benchmark/TL/TL-2-2.fsp"));
        ControllerSpecDef spec = model.controllerSpecs().stream()
            .filter(ControllerSpecDef::nonblocking).reduce((a,b)->b).get();
        List<LtlPropertyDef> safety = new ArrayList<>();
        for (String n : spec.safety())
            model.ltlProperties().stream().filter(p->p.name().equals(n)).findFirst().ifPresent(safety::add);

        Heuristic h = new Heuristic() {
            int step = 0;
            String[] script = {"get[0]", "put[1]"};
            public int pick(List<ExtendedTransition> pending) {
                System.out.println("Step " + (step+1) + " frontier (" + pending.size() + "):");
                for (ExtendedTransition t : pending)
                    System.out.println("  " + t.from().substring(0, Math.min(50, t.from().length()))
                        + " --[" + t.action() + "]--> " + t.to().substring(0, Math.min(20,t.to().length())));
                if (step < script.length) {
                    String want = script[step++];
                    for (int i = 0; i < pending.size(); i++)
                        if (pending.get(i).action().equals(want)) return i;
                    System.out.println("  ** scripted action '" + want + "' NOT FOUND, fallback");
                }
                return 0;
            }
        };
        new OTFDirectedControledSyntesisNonBlocking(
            new ArrayList<>(model.processes()), safety,
            new HashSet<>(spec.marking()), new HashSet<>(spec.controllable()), h, false, 2
        ).run();
    }
}
