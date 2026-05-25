package newmtsa.synthesis.heuristics;

import newmtsa.synthesis.ExtendedTransition;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

public class HumanHeuristic implements Heuristic {

    private final Scanner scanner;
    private final Set<String> hidden = new HashSet<>();
    private final Map<String, Integer> marked = new LinkedHashMap<>();
    private int markCounter = 1;

    public HumanHeuristic() {
        this.scanner = new Scanner(System.in);
    }

    @Override
    public int pick(List<ExtendedTransition> pending) {
        printFrontier(pending);

        while (true) {
            System.out.print("Choose index (or 'del i'/'del i-j'/'reset'/'mark i'): ");
            String line = scanner.nextLine().trim();

            if (line.equalsIgnoreCase("reset")) {
                hidden.clear();
                System.out.println("  Hidden transitions cleared.");
                printFrontier(pending);
                continue;
            }

            if (line.toLowerCase().startsWith("del ")) {
                handleDel(line.substring(4).trim(), pending);
                printFrontier(pending);
                continue;
            }

            if (line.toLowerCase().startsWith("mark ")) {
                handleMark(line.substring(5).trim(), pending);
                printFrontier(pending);
                continue;
            }

            try {
                int choice = Integer.parseInt(line);
                if (choice < 0 || choice >= pending.size()) {
                    System.out.println("  Out of range. Try again.");
                } else if (hidden.contains(key(pending.get(choice)))) {
                    System.out.println("  Transition " + choice + " is hidden. Choose a visible index.");
                } else {
                    marked.remove(key(pending.get(choice)));
                    return choice;
                }
            } catch (NumberFormatException e) {
                System.out.println("  Unknown command. Enter an index, 'del i', 'del i-j', 'reset', or 'mark i'.");
            }
        }
    }

    private void printFrontier(List<ExtendedTransition> pending) {
        long visible = pending.stream().filter(t -> !hidden.contains(key(t))).count();
        System.out.println("[HumanHeuristic] Frontier: " + pending.size() + " total, " + visible + " visible:");
        for (int i = 0; i < pending.size(); i++) {
            ExtendedTransition t = pending.get(i);
            if (hidden.contains(key(t))) continue;
            Integer markLevel = marked.get(key(t));
            String stars = markLevel != null ? " [34m" + "*".repeat(markLevel) + "[0m" : "";
            System.out.printf("  [%d] (step %d) %s --[%s]--> %s%s%n", i, t.step(), t.from(), t.action(), t.to(), stars);
        }
    }

    private void handleDel(String range, List<ExtendedTransition> pending) {
        String[] parts = range.split("-");
        try {
            if (parts.length == 1) {
                int idx = Integer.parseInt(parts[0].trim());
                if (idx < 0 || idx >= pending.size()) {
                    System.out.println("  Index out of bounds [0-" + (pending.size() - 1) + "].");
                    return;
                }
                hidden.add(key(pending.get(idx)));
                System.out.println("  Hidden transition " + idx + ".");
            } else if (parts.length == 2) {
                int lo = Integer.parseInt(parts[0].trim());
                int hi = Integer.parseInt(parts[1].trim());
                if (lo > hi || lo < 0 || hi >= pending.size()) {
                    System.out.println("  Range out of bounds [0-" + (pending.size() - 1) + "].");
                    return;
                }
                for (int i = lo; i <= hi; i++) hidden.add(key(pending.get(i)));
                System.out.println("  Hidden transitions " + lo + "-" + hi + ".");
            } else {
                System.out.println("  Bad format. Use: del i  or  del i-j");
            }
        } catch (NumberFormatException e) {
            System.out.println("  Bad format. Use: del i  or  del i-j");
        }
    }

    private void handleMark(String indexStr, List<ExtendedTransition> pending) {
        try {
            int idx = Integer.parseInt(indexStr);
            if (idx < 0 || idx >= pending.size()) {
                System.out.println("  Index out of bounds [0-" + (pending.size() - 1) + "].");
                return;
            }
            String k = key(pending.get(idx));
            if (!marked.containsKey(k)) {
                marked.put(k, markCounter++);
            }
            System.out.println("  Marked transition " + idx + " with level " + marked.get(k) + ".");
        } catch (NumberFormatException e) {
            System.out.println("  Bad format. Use: mark i");
        }
    }

    private static String key(ExtendedTransition t) {
        return t.from() + "||" + t.action() + "||" + t.to();
    }
}
