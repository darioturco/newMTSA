package newmtsa.synthesis.heuristics;

import newmtsa.synthesis.ExtendedTransition;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * BFS-order heuristic: expands transitions layer by layer, imitating
 * breadth-first exploration of the state space.
 *
 * <p>Transitions are enqueued in FIFO order as they first appear in the
 * pending frontier, so all transitions at depth <em>d</em> are processed
 * before any transition at depth <em>d+1</em>.
 */
public class BFSHeuristic implements Heuristic {

    /** FIFO queue of transitions in BFS order. */
    private final Deque<ExtendedTransition> queue = new ArrayDeque<>();
    /** Tracks transitions already enqueued to avoid duplicates. */
    private final Set<ExtendedTransition> seen = new HashSet<>();

    @Override
    public int pick(List<ExtendedTransition> pending) {
        for (ExtendedTransition t : pending) {
            if (seen.add(t)) queue.add(t);
        }

        ExtendedTransition head;
        while ((head = queue.poll()) != null) {
            int idx = pending.indexOf(head);
            if (idx >= 0) return idx;
        }

        return 0;
    }
}
