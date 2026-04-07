package newmtsa.parser.ast;

import java.util.List;

public record ControllerSpecDef(
        String name,
        List<String> liveness,
        List<String> safety,
        List<String> assumption,
        List<String> controllable,
        List<String> marking,
        boolean nonblocking
) {}
