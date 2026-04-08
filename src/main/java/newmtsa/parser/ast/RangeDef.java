package newmtsa.parser.ast;

/**
 * A range declaration: {@code range Phil = 0..Philosophers-1}
 *
 * <p>Both bounds are fully resolved to integers at parse time (constants are
 * substituted, arithmetic is evaluated).  The compact constructor enforces
 * {@code init <= end}; violating this throws {@link IllegalArgumentException}.
 */
public record RangeDef(String name, int init, int end) {
    public RangeDef {
        if (init > end)
            throw new IllegalArgumentException(
                    "Range '" + name + "': init (" + init + ") must be <= end (" + end + ")");
    }

    /** Number of distinct values in the range (inclusive on both ends). */
    public int size() { return end - init + 1; }
}
