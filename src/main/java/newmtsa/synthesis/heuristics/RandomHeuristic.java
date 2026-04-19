package newmtsa.synthesis.heuristics;

import newmtsa.synthesis.ExtendedTransition;

import java.util.List;
import java.util.Random;

/**
 * Selects the next transition from the pending frontier.
 *
 * <p>When constructed with a {@code script} list of indices, the i-th call to
 * {@link #pick} uses {@code script.get(i)} as the frontier index to select.
 * Once the script is exhausted, or if the scripted index is out of bounds for
 * the current frontier, selection falls back to uniform random (printing a
 * warning in the out-of-bounds case).
 */
public class RandomHeuristic implements Heuristic {

    private final Random     rng;
    private final List<Integer> script;
    private int              step = 0;

    public RandomHeuristic() {
        this.rng    = new Random();
        this.script = List.of();
    }

    public RandomHeuristic(long seed) {
        this.rng    = new Random(seed);
        this.script = List.of();
    }

    /**
     * @param script ordered list of frontier indices to use at each step;
     *               random selection takes over once the list is exhausted
     */
    public RandomHeuristic(List<Integer> script) {
        this.rng    = new Random();
        this.script = List.copyOf(script);
    }

    /**
     * @param seed   RNG seed for the random fallback
     * @param script ordered list of frontier indices to use at each step
     */
    public RandomHeuristic(long seed, List<Integer> script) {
        this.rng    = new Random(seed);
        this.script = List.copyOf(script);
    }

    @Override
    public ExtendedTransition pick(List<ExtendedTransition> pending) {
        if (step < script.size()) {
            int idx = script.get(step++);
            if (idx >= 0 && idx < pending.size()) {
                return pending.get(idx);
            }
            System.err.println("[RandomHeuristic] WARNING: scripted index " + idx
                    + " is out of bounds for frontier of size " + pending.size()
                    + " at step " + (step - 1) + " — falling back to random.");
        } else {
            step++;
        }
        return pending.get(rng.nextInt(pending.size()));
    }
}
