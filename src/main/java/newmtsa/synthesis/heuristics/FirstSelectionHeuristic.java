package newmtsa.synthesis.heuristics;

import newmtsa.synthesis.ExtendedTransition;
import java.util.List;

/** Always selects the first action from the frontier of pending transitions. */
public class FirstSelectionHeuristic implements Heuristic {

    @Override
    public int pick(List<ExtendedTransition> pending) {
        return 0;
    }
}
