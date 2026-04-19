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
    public ExtendedTransition pick(List<ExtendedTransition> pending) {
        // Enqueue any transitions that have not been seen yet, preserving
        // the insertion order in which they appear in pending.
        for (ExtendedTransition t : pending) {
            if (seen.add(t)) {
                queue.add(t);
            }
        }

        // Return the front of the queue that is still present in pending.
        // (Items are removed from pending by the caller after pick returns,
        // so the only reason the head might be absent is if the same
        // ExtendedTransition object appeared more than once and was already
        // consumed — handled by the seen set above.)
        ExtendedTransition head;
        while ((head = queue.poll()) != null) {
            if (pending.contains(head)) {
                return head;
            }
        }

        // Should never be reached when called with a non-empty pending list.
        return pending.get(0);
    }
}
