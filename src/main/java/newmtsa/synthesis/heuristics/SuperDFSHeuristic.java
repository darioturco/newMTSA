package newmtsa.synthesis.heuristics;

import newmtsa.synthesis.Director;
import newmtsa.synthesis.ExtendedTransition;

import java.util.*;

/**
 * SuperDFS heuristic: depth-first search guided by a stack (pile) and an ignore list.
 *
 * <p>Rules applied at each state:
 * <ol>
 *   <li>If the state has noncontrollable transitions: pick one, push the rest onto the stack.</li>
 *   <li>If the state has exactly one non-error controllable transition: pick it.</li>
 *   <li>If the state has multiple non-error controllable transitions: delegate to the
 *       family-specific picker; push the rest onto the ignore list and record the decision.</li>
 * </ol>
 * When the current DFS branch is exhausted (no transitions from the current state in the
 * frontier), the next transition is taken from the stack.  When the stack is also empty,
 * transitions are taken from the ignore list.
 *
 * <p>At the end of guided exploration the full decision log is printed.
 */
public class SuperDFSHeuristic implements Heuristic {

    private final String family;
    private final int    n;
    private final int    k;

    private SynthesisContext ctx;
    private Set<String>      controllable;

    private String currentState = null;

    private final Deque<ExtendedTransition>  stack          = new ArrayDeque<>();
    private final List<ExtendedTransition>   ignored        = new ArrayList<>();
    private final Set<String>                decidedStates  = new HashSet<>();

    private record ChoiceRecord(int step, ExtendedTransition chosen, List<ExtendedTransition> skipped) {}
    private final List<ChoiceRecord> choices = new ArrayList<>();

    private int     stepCount         = 0;
    private int     ignoredExpansions = 0;
    private boolean summaryPrinted    = false;
    private boolean forceStackPop     = false;

    public SuperDFSHeuristic() {
        this("unknown", 0, 0);
    }

    private final int safePlace;

    public SuperDFSHeuristic(String family, int n, int k) {
        this.family    = family;
        this.n         = n;
        this.k         = k;
        this.safePlace = (2 * k + 1) / 2;
    }

    @Override
    public void init(SynthesisContext ctx) {
        this.ctx          = ctx;
        this.controllable = ctx.controllable();
    }

    /**
     * Selects the next transition to expand using depth-first order guided by the stack and ignore list.
     * Noncontrollable transitions are always preferred; remaining ones are pushed onto the stack.
     * Among controllable transitions the family-specific picker is used; unchosen ones go to the ignore list.
     * When no transition from the current state exists in the frontier, falls back to stack then ignore list.
     */
    @Override
    public int pick(List<ExtendedTransition> pending) {
        stepCount++;

        if (forceStackPop) {
            forceStackPop = false;
            return pickFromStackOrIgnored(pending);
        }

        if (currentState == null && !pending.isEmpty()) {
            currentState = pending.get(0).from();
        }

        List<ExtendedTransition> nonCtrl = new ArrayList<>();
        List<ExtendedTransition> ctrl    = new ArrayList<>();

        for (ExtendedTransition t : pending) {
            if (!t.from().equals(currentState)) continue;
            if (t.isControllable(controllable)) {
                if (!goesToError(t)) ctrl.add(t);
            } else {
                nonCtrl.add(t);
            }
        }

        ExtendedTransition chosen = null;

        if (!nonCtrl.isEmpty()) {
            chosen = nonCtrl.get(0);
            for (int i = nonCtrl.size() - 1; i >= 1; i--) {
                stack.push(nonCtrl.get(i));
            }
        } else if (!ctrl.isEmpty()) {
            if (decidedStates.contains(currentState)) {
                return pickFromStackOrIgnored(pending);
            }
            chosen = pickControllable(ctrl);
            if (ctrl.size() > 1) {
                ExtendedTransition finalChosen = chosen;
                List<ExtendedTransition> others = new ArrayList<>();
                for (ExtendedTransition t : ctrl) {
                    if (t != finalChosen) others.add(t);
                }
                ignored.addAll(others);
                choices.add(new ChoiceRecord(stepCount, chosen, others));
                decidedStates.add(currentState);
            }
        }

        if (chosen != null) {
            int idx = findInPending(pending, chosen);
            if (idx >= 0) {
                currentState = chosen.to();
                if (ctx.getFutureAddToFrontier(chosen).isEmpty()) forceStackPop = true;
                return idx;
            }
        }

        return pickFromStackOrIgnored(pending);
    }

    // ── family-specific controllable picker ───────────────────────────────────

    private ExtendedTransition pickControllable(List<ExtendedTransition> ctrl) {
        return switch (family) {
            case "AT" -> atPickControllable(ctrl);
            case "DP" -> dpPickControllable(ctrl);
            case "TL" -> tlPickControllable(ctrl);
            case "BW" -> bwPickControllable(ctrl);
            case "CM" -> cmPickControllable(ctrl);
            case "TA" -> taPickControllable(ctrl);
            default   -> ctrl.get(0);
        };
    }

    /**
     * DP: state layout is blocks of 3 per philosopher at positions 3i, 3i+1, 3i+2.
     * Position 3i = philosopher sub-state ("Hungry", "Ready", "Thinking", …).
     * Priority: pick {@code take[i][j]} where philosopher i is "Ready" (min i);
     * if none, pick where philosopher i is "Hungry" (min i).
     */
    private ExtendedTransition dpPickControllable(List<ExtendedTransition> ctrl) {
        int                readyI = Integer.MAX_VALUE;
        ExtendedTransition ready  = null;
        int                hungryI = Integer.MAX_VALUE;
        ExtendedTransition hungry  = null;

        for (ExtendedTransition t : ctrl) {
            if (!t.action().startsWith("take[")) continue;
            int      i     = t.extractFirstIndex();
            String[] parts = t.from().split("\\|");
            if (3 * i >= parts.length) continue;
            String philoState   = parts[3 * i];
            String monitorState = (3 * i + 2 < parts.length) ? parts[3 * i + 2] : "";
            if ("Ready".equals(philoState) && i < readyI) {
                readyI = i;
                ready  = t;
            } else if ("Hungry".equals(philoState) && !"Done".equals(monitorState) && i < hungryI) {
                hungryI = i;
                hungry  = t;
            }
        }
        if (ready  != null) return ready;
        if (hungry != null) return hungry;
        return ctrl.get(0);
    }

    private ExtendedTransition tlPickControllable(List<ExtendedTransition> ctrl) {
        Set<String> explored = (ctx != null) ? ctx.exploredStates() : Set.of();

        for (ExtendedTransition t : ctrl) {
            if (t.action().startsWith("get[") && explored.contains(t.to())) return t;
        }

        int                bestI = -1;
        ExtendedTransition best  = null;
        for (ExtendedTransition t : ctrl) {
            if (!t.action().startsWith("get[")) continue;
            int i = t.extractFirstIndex();
            if (!goesToError(t) && i > bestI) {
                bestI = i;
                best  = t;
            }
        }
        return best != null ? best : ctrl.get(0);
    }

    private ExtendedTransition bwPickControllable(List<ExtendedTransition> ctrl) {
        // Rule 1: approve not leading to error
        for (ExtendedTransition t : ctrl) {
            if ("approve".equals(t.action()) && !goesToError(t)) return t;
        }

        // Rule 2: refuse where to-state first substate = 'Rejected'
        for (ExtendedTransition t : ctrl) {
            if ("refuse".equals(t.action())) {
                String[] toParts = t.to().split("\\|");
                if (toParts.length > 0 && "Rejected".equals(toParts[0])) return t;
            }
        }

        // Rule 3: assign[i] with min i where crew i is 'Pending' and no error
        int                bestI = Integer.MAX_VALUE;
        ExtendedTransition best  = null;
        for (ExtendedTransition t : ctrl) {
            if (!t.action().startsWith("assign[")) continue;
            if (goesToError(t)) continue;
            int i = t.extractFirstIndex();
            String[] fromParts = t.from().split("\\|");
            if (i + 1 < fromParts.length && "Pending".equals(fromParts[i + 1]) && i < bestI) {
                bestI = i;
                best  = t;
            }
        }
        return best != null ? best : ctrl.get(0);
    }

    private ExtendedTransition cmPickControllable(List<ExtendedTransition> ctrl) {
        int                bestDist = Integer.MAX_VALUE;
        int                bestI    = Integer.MAX_VALUE;
        ExtendedTransition best     = null;

        for (ExtendedTransition t : ctrl) {
            String a = t.action();
            if (!a.startsWith("mouse[")) continue;
            int mouseI = t.extractFirstIndex();
            int moveJ  = extractMovePosition(a);
            int dist   = Math.abs(moveJ - safePlace);

            if (dist < bestDist || (dist == bestDist && mouseI < bestI)) {
                bestDist = dist;
                bestI    = mouseI;
                best     = t;
            }
        }
        return best != null ? best : ctrl.get(0);
    }

    private boolean heightsAreConsecutive(String fromState) {
        String[] parts = fromState.split("\\|");
        boolean seenEmpty = false;
        for (int h = 0; h < k; h++) {
            String slot = parts[2 + h];
            if ("Empty".equals(slot)) {
                seenEmpty = true;
            } else if (slot.startsWith("Occupied[") && seenEmpty) {
                return false;
            }
        }
        return true;
    }

    private int extractMovePosition(String action) {
        int lastBracket  = action.lastIndexOf('[');
        int closeBracket = action.lastIndexOf(']');
        return Integer.parseInt(action.substring(lastBracket + 1, closeBracket));
    }

    private ExtendedTransition taPickControllable(List<ExtendedTransition> ctrl) {
        return ctrl.get(0);
    }

    /**
     * AT: prefer approach[i] if present; otherwise pick descend[i][j] where:
     *   1. plane i is currently at height j+1 (one step above target — no skipping)
     *   2. height j is Empty in the from-state
     * Among valid candidates, pick the one with lowest j.
     */
    private ExtendedTransition atPickControllable(List<ExtendedTransition> ctrl) {
        for (ExtendedTransition t : ctrl) {
            if (t.action().startsWith("approach[") && heightsAreConsecutive(t.from())) return t;
        }

        int                bestJ = Integer.MAX_VALUE;
        ExtendedTransition best  = null;
        for (ExtendedTransition t : ctrl) {
            String a = t.action();
            if (!a.startsWith("descend[")) continue;
            int      j     = descendY(a);
            String[] parts = t.from().split("\\|");
            if (parts.length > j + 2 && "Empty".equals(parts[j + 2]) && j < bestJ) {
                bestJ = j;
                best  = t;
            }
        }
        return best != null ? best : ctrl.get(0);
    }

    // ── backtrack ─────────────────────────────────────────────────────────────

    @Override
    public void notifyExplorationEnd(Director result) {
        printSummary(result);
    }

    private int pickFromStackOrIgnored(List<ExtendedTransition> pending) {
        while (!stack.isEmpty()) {
            ExtendedTransition t = stack.pop();
            int idx = findInPending(pending, t);
            if (idx >= 0) {
                currentState = t.to();
                return idx;
            }
        }

        Iterator<ExtendedTransition> it = ignored.iterator();
        while (it.hasNext()) {
            ExtendedTransition t = it.next();
            it.remove();
            int idx = findInPending(pending, t);
            if (idx >= 0) {
                ignoredExpansions++;
                currentState = t.to();
                return idx;
            }
        }

        if (!pending.isEmpty()) {
            currentState = pending.get(0).to();
            return 0;
        }
        return 0;
    }

    // ── summary ───────────────────────────────────────────────────────────────

    private void printSummary(Director result) {
        if (summaryPrinted) return;
        summaryPrinted = true;

        System.out.println("\n=== SuperDFS Exploration Summary ===");
        if (choices.isEmpty()) {
            System.out.println("  No multi-controllable choice points.");
        } else {
            System.out.println("  Multi-controllable decisions (" + choices.size() + "):");
            for (ChoiceRecord cr : choices) {
                boolean chosenInDirector = inDirector(result, cr.chosen());
                System.out.println("  Step " + cr.step() + ":"
                        + (result.isRealizable() ? " [chosen was " + (chosenInDirector ? "RIGHT" : "WRONG") + "]" : ""));
                System.out.println("    Chosen:  " + cr.chosen().format());
                for (ExtendedTransition sk : cr.skipped()) {
                    boolean skInDirector = inDirector(result, sk);
                    String tag = result.isRealizable() ? (skInDirector ? " [IN DIRECTOR]" : "") : "";
                    System.out.println("    Ignored: " + sk.format() + tag);
                }
            }
        }
        if (result.isRealizable() && !choices.isEmpty()) {
            long right = choices.stream().filter(cr -> inDirector(result, cr.chosen())).count();
            long wrong = choices.size() - right;
            System.out.println("  RIGHT decisions: " + right + " / " + choices.size()
                    + "   WRONG decisions: " + wrong + " / " + choices.size());
        }
        System.out.println("  Expansions from ignored list: " + ignoredExpansions);
        System.out.println("====================================\n");
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    private boolean inDirector(Director d, ExtendedTransition t) {
        return d.goals().contains(t.to());
    }

    private boolean goesToError(ExtendedTransition t) {
        if (ctx != null && ctx.errors().contains(t.to())) return true;
        for (String part : t.to().split("\\|")) {
            if ("ERROR".equals(part)) return true;
        }
        return false;
    }

    /** Extracts the height index j from {@code descend[i][j]}. */
    private int descendY(String action) {
        int b2 = action.indexOf(']');
        int b3 = action.indexOf('[', b2);
        int b4 = action.indexOf(']', b3);
        return Integer.parseInt(action.substring(b3 + 1, b4));
    }

    private int findInPending(List<ExtendedTransition> pending, ExtendedTransition t) {
        for (int i = 0; i < pending.size(); i++) {
            ExtendedTransition p = pending.get(i);
            if (p.from().equals(t.from()) && p.action().equals(t.action()) && p.to().equals(t.to())) {
                return i;
            }
        }
        return -1;
    }
}
