package newmtsa.synthesis.heuristics;

import newmtsa.synthesis.ExtendedTransition;

import java.util.List;

/** Always selects the first action from the frontier of pending transitions. */
public class FirstSelectionHeuristic implements Heuristic {

    @Override
    public ExtendedTransition pick(List<ExtendedTransition> pending) {
        return pending.get(0);
    }
}
