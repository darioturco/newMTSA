package newmtsa.synthesis.heuristics;

import newmtsa.synthesis.ExtendedTransition;

import java.util.List;
import java.util.Scanner;

/**
 * Interactive heuristic that prints the frontier and asks a human to pick
 * which transition to expand next.
 */
public class HumanHeuristic implements Heuristic {

    private final Scanner scanner;

    public HumanHeuristic() {
        this.scanner = new Scanner(System.in);
    }

    @Override
    public int pick(List<ExtendedTransition> pending) {
        System.out.println("[HumanHeuristic] Current frontier (" + pending.size() + " transitions):");
        for (int i = 0; i < pending.size(); i++) {
            ExtendedTransition t = pending.get(i);
            System.out.printf("  [%d] %s --[%s]--> %s%n", i, t.from(), t.action(), t.to());
        }

        int choice = -1;
        while (choice < 0 || choice >= pending.size()) {
            System.out.print("Choose transition index [0-" + (pending.size() - 1) + "]: ");
            String line = scanner.nextLine().trim();
            try {
                choice = Integer.parseInt(line);
                if (choice < 0 || choice >= pending.size()) {
                    System.out.println("  Out of range. Try again.");
                }
            } catch (NumberFormatException e) {
                System.out.println("  Not a number. Try again.");
            }
        }
        return choice;
    }
}
