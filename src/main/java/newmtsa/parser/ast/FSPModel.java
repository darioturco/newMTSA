package newmtsa.parser.ast;

import java.util.List;

public record FSPModel(
        List<ProcessDef> processes,
        List<CompositeDef> composites,
        List<FluentDef> fluents,
        List<AssertDef> asserts,
        List<LtlPropertyDef> ltlProperties,
        List<SetDef> sets,
        List<ControllerSpecDef> controllerSpecs,
        List<String> errors
) {}
