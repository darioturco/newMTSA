package newmtsa.synthesis.heuristics;

import newmtsa.synthesis.ExtendedTransition;

import java.util.*;

/**
 * SuperHuman heuristic: identical to SuperDFS except that when a controllable-choice
 * decision must be made the human user is prompted interactively.
 *
 * In Case 5 (mixed noncontrollable + controllable) the base class passes
 * canReturnNull=true; returning null lets SuperHeuristic expand the noncontrollable
 * transition as the default behavior.
 */
public class SuperHumanHeuristic extends SuperHeuristic {

    private final Scanner scanner = new Scanner(System.in);

    @Override
    protected ExtendedTransition pickControllable(List<ExtendedTransition> ctrl, boolean canReturnNull) {
        if (canReturnNull) return null;

        System.out.println();
        System.out.println("[SuperHuman] Controllable decision at state:");
        System.out.println("  " + currentState);
        System.out.println("  Options:");
        for (int i = 0; i < ctrl.size(); i++) {
            ExtendedTransition t = ctrl.get(i);
            System.out.printf("    [%d]  --%s-->  %s%n", i, t.action(), t.to());
        }

        while (true) {
            System.out.print("  Choose (index): ");
            String line = scanner.nextLine().trim();
            try {
                int idx = Integer.parseInt(line);
                if (idx >= 0 && idx < ctrl.size()) return ctrl.get(idx);
                System.out.println("  Out of range [0-" + (ctrl.size() - 1) + "]. Try again.");
            } catch (NumberFormatException e) {
                System.out.println("  Enter a number.");
            }
        }
    }
}
