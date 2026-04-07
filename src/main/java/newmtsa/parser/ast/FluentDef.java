package newmtsa.parser.ast;

import java.util.List;

public record FluentDef(
        String name,
        List<String> initiatingActions,
        List<String> terminatingActions
) {}
