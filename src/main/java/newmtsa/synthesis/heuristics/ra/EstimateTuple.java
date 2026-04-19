package newmtsa.synthesis.heuristics.ra;

/**
 * A single per-component estimate tuple {@code ⟨m, d⟩} used by the RA heuristic.
 *
 * <p>The {@code m} field encodes confidence in reaching a marked state:
 * <ul>
 *   <li>{@code m = -1} — path leads to a state in <em>Goals</em> (RA.G only)</li>
 *   <li>{@code m =  0} — path leads to an already-<em>visited</em> marked state</li>
 *   <li>{@code m =  1} — path leads to an <em>unvisited</em> marked state</li>
 *   <li>{@code m =  2} — no marked state is reachable ({@link #UNREACHABLE})</li>
 * </ul>
 * {@code d} is the estimated step count to the nearest such state.
 *
 * <p>The natural ordering is lexicographic on {@code (m, d)}: smaller is better
 * (closer to a known-good target).
 */
public record EstimateTuple(int m, int d) implements Comparable<EstimateTuple> {

    /** Sentinel value indicating no marked state is reachable from this component. */
    public static final EstimateTuple UNREACHABLE = new EstimateTuple(2, Integer.MAX_VALUE);

    @Override
    public int compareTo(EstimateTuple o) {
        int cm = Integer.compare(this.m, o.m);
        return cm != 0 ? cm : Integer.compare(this.d, o.d);
    }
}
