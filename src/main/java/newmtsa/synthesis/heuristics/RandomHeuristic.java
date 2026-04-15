package newmtsa.synthesis.heuristics;

import newmtsa.synthesis.ExtendedTransition;

import java.util.List;
import java.util.Random;

/** Selects the next transition uniformly at random from the pending frontier. */
public class RandomHeuristic implements Heuristic {
    private final Random rng;

    public RandomHeuristic() { this.rng = new Random(); }
    public RandomHeuristic(long seed) { this.rng = new Random(seed); }

    @Override
    public ExtendedTransition pick(List<ExtendedTransition> pending) {
        return pending.get(rng.nextInt(pending.size()));
    }
}
