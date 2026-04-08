package newmtsa.parser.ast;

import java.util.List;
import java.util.Set;

public record FSPModel(
        List<LTS> processes,
        List<ParallelCompositionLazy> composites,
        List<FluentDef> fluents,
        List<AssertDef> asserts,
        List<LtlPropertyDef> ltlProperties,
        List<SetDef> sets,
        List<ControllerSpecDef> controllerSpecs,
        List<ConstDef> constants,
        List<RangeDef> ranges,
        List<MacroDef> macros,
        Set<String> actions,
        List<String> errors
) {}
