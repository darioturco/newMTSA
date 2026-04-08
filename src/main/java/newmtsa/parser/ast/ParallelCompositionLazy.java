package newmtsa.parser.ast;

import java.util.List;

/**
 * A lazy parallel composition of two or more LTS.
 *
 * <p>The {@code ||} operator in FSP computes the parallel composition of multiple
 * LTS, producing a new (potentially much larger) LTS.  Because this operation is
 * expensive, it is deferred: the operands are stored and the composition is only
 * executed when {@link #execute()} is called.
 *
 * <ul>
 *   <li>{@code name}       — name of the resulting composed LTS</li>
 *   <li>{@code components} — the LTS operands in declaration order</li>
 * </ul>
 *
 * <p><b>Current stub:</b> {@link #execute()} returns the first component unchanged.
 * The real parallel-composition algorithm will be implemented here later.
 */
public record ParallelCompositionLazy(String name, List<LTS> components) {

    /** Execute the parallel composition and return the resulting LTS. */
    public LTS execute() {
        if (components.isEmpty())
            throw new IllegalStateException(
                    "ParallelCompositionLazy '" + name + "' has no components to compose.");
        return components.get(0);
    }
}
