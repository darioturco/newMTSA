package newmtsa.synthesis.heuristics;

import newmtsa.synthesis.Director;
import newmtsa.synthesis.ExtendedTransition;

import java.util.*;
import java.util.Comparator;
import java.util.function.Predicate;
import java.util.function.ToIntFunction;

/**
 * SuperDFS heuristic: depth-first search guided by a stack (pile) and an ignore list.
 * Extends SuperHeuristic, implementing pickControllable() with family-specific logic.
 */
public class SuperDFSHeuristic extends SuperHeuristic {

    private final String family;
    private final int    n;
    private final int    k;
    private final int    safePlace;

    private record ChoiceRecord(int step, ExtendedTransition chosen, List<ExtendedTransition> skipped) {}
    private final List<ChoiceRecord> choices = new ArrayList<>();

    private int subHeuristicCalls = 0;

    public SuperDFSHeuristic() {
        this("unknown", 0, 0);
    }

    public SuperDFSHeuristic(String family, int n, int k) {
        this.family    = family;
        this.n         = n;
        this.k         = k;
        this.safePlace = (2 * k + 1) / 2;
    }

    @Override
    protected ExtendedTransition pickControllable(List<ExtendedTransition> ctrl, boolean canReturnNull) {
        subHeuristicCalls++;
        return switch (family) {
            case "AT" -> atPickControllable(ctrl, canReturnNull);
            case "DP" -> dpPickControllable(ctrl, canReturnNull);
            case "TL" -> tlPickControllable(ctrl, canReturnNull);
            case "BW" -> bwPickControllable(ctrl, canReturnNull);
            case "CM" -> cmPickControllable(ctrl, canReturnNull);
            case "TA" -> taPickControllable(ctrl, canReturnNull);
            default   -> ctrl.get(0);
        };
    }

    @Override
    protected void onControllableChosen(ExtendedTransition chosen, List<ExtendedTransition> others) {
        choices.add(new ChoiceRecord(stepCount, chosen, others));
    }

    @Override
    public void notifyExplorationEnd(Director result) {
        System.out.println("[SuperDFS] Sub-heuristic calls: " + subHeuristicCalls);
    }

    // ── family-specific controllable pickers ──────────────────────────────────

    /**
     * DP: state layout is blocks of 3 per philosopher at positions 3i, 3i+1, 3i+2.
     * Position 3i = philosopher sub-state ("Hungry", "Ready", "Thinking", …).
     * Priority: pick {@code take[i][j]} where philosopher i is "Ready" (min i);
     * if none, pick where philosopher i is "Hungry" (min i).
     */
    private ExtendedTransition dpPickControllable(List<ExtendedTransition> ctrl, boolean canReturnNull) {
        if (canReturnNull) return null;
        ExtendedTransition ready = minByKey(ctrl, "take[",
            t -> { int i = t.extractFirstIndex(); String[] p = t.from().split("\\|"); return 3*i < p.length && "Ready".equals(p[3*i]); },
            ExtendedTransition::extractFirstIndex);
        if (ready != null) return ready;
        ExtendedTransition hungry = minByKey(ctrl, "take[",
            t -> { int i = t.extractFirstIndex(); String[] p = t.from().split("\\|"); if (3*i >= p.length) return false; String ms = (3*i+2 < p.length) ? p[3*i+2] : ""; return "Hungry".equals(p[3*i]) && !"Done".equals(ms); },
            ExtendedTransition::extractFirstIndex);
        if (hungry != null) return hungry;
        return ctrl.get(0);
    }

    /**
     * TL: candidates are {@code get[i]} (pick up piece at position i).
     * Priority: (1) get[i] whose destination was already explored (closes the cycle);
     * (2) get[i] with max i that does not go to error (advance as far as possible);
     * (3) fallback: first transition.
     */
    private ExtendedTransition tlPickControllable(List<ExtendedTransition> ctrl, boolean canReturnNull) {
        if (canReturnNull) return null;
        Set<String> explored = (ctx != null) ? ctx.exploredStates() : Set.of();
        for (ExtendedTransition t : ctrl) {
            if (t.action().startsWith("get[") && explored.contains(t.to())) return t;
        }
        ExtendedTransition best = maxByKey(ctrl, "get[", t -> !goesToError(t), ExtendedTransition::extractFirstIndex);
        return best != null ? best : ctrl.get(0);
    }

    /**
     * BW: candidates are {@code approve}, {@code refuse}, {@code assign[i]}.
     * Priority: (1) approve if it does not go to error;
     * (2) refuse if the first sub-state of the from-state is "Rejected";
     * (3) assign[i] with min i where crew i is "Pending" or "Rejected[x]" and no error;
     * (4) fallback: first transition.
     */
    private ExtendedTransition bwPickControllable(List<ExtendedTransition> ctrl, boolean canReturnNull) {
        if (canReturnNull) return null;
        for (ExtendedTransition t : ctrl) {
            if ("approve".equals(t.action()) && !goesToError(t)) return t;
        }
        for (ExtendedTransition t : ctrl) {
            if ("refuse".equals(t.action())) {
                String[] fromParts = t.from().split("\\|");
                if (fromParts.length > 0 && "Rejected".equals(fromParts[0])) return t;
            }
        }
        ExtendedTransition best = minByKey(ctrl, "assign[",
            t -> { if (goesToError(t)) return false; int i = t.extractFirstIndex(); String[] p = t.from().split("\\|"); String cs = i+1 < p.length ? p[i+1] : ""; return "Pending".equals(cs) || cs.startsWith("Rejected"); },
            ExtendedTransition::extractFirstIndex);
        return best != null ? best : ctrl.get(0);
    }

    /**
     * CM: candidates are {@code mouse[i][move[j]]} (mouse i moves to position j).
     * Pick the move that minimises |j - safePlace| (closest to the safe centre position).
     * Tie-break: lowest mouse index i. Fallback: first transition.
     * safePlace = floor((2k+1)/2).
     */
    private ExtendedTransition cmPickControllable(List<ExtendedTransition> ctrl, boolean canReturnNull) {
        if (canReturnNull) return null;
        ExtendedTransition best = minByComparator(ctrl, "mouse[", (a, b) -> {
            int distA = Math.abs(a.extractLastIndex() - safePlace);
            int distB = Math.abs(b.extractLastIndex() - safePlace);
            int cmp   = Integer.compare(distA, distB);
            return cmp != 0 ? cmp : Integer.compare(a.extractFirstIndex(), b.extractFirstIndex());
        });
        return best != null ? best : ctrl.get(0);
    }

    /**
     * TA: state layout is [Agency | AgencyMonitor | Service(0) | ServiceMonitor(0) | ... | Service(n-1) | ServiceMonitor(n-1) | fluent].
     * ServiceMonitor(i) is at parts[3 + 2*i]. Controllable actions: agency.succ, agency.fail, purchase[i], cancel[i].
     * Priority:
     * (1) All ServiceMonitor substates are "Success" → expand agency.succ (not going to error).
     * (2) agency.fail does not lead to any ERROR substate → expand agency.fail.
     * (3) Otherwise → expand purchase[i] where ServiceMonitor(i) is not "Success"
     *     and i is closest to j parsed from AgencyMonitor substate "Disallow[j]".
     */
    private ExtendedTransition taPickControllable(List<ExtendedTransition> ctrl, boolean canReturnNull) {
        if (canReturnNull) return null;

        String[] parts = ctrl.get(0).from().split("\\|");

        boolean allSuccess = true;
        for (int i = 0; i < n; i++) {
            int idx = 3 + 2 * i;
            if (idx >= parts.length || !"Success".equals(parts[idx])) { allSuccess = false; break; }
        }

        if (allSuccess) {
            for (ExtendedTransition t : ctrl) {
                if ("agency.succ".equals(t.action()) && !goesToError(t)) return t;
            }
        }

        for (ExtendedTransition t : ctrl) {
            if ("agency.fail".equals(t.action()) && !goesToError(t)) return t;
        }

        int j = 0;
        String agentMonState = parts.length > 1 ? parts[1] : "";
        if (agentMonState.startsWith("Disallow[")) {
            int b1 = agentMonState.indexOf('[');
            int b2 = agentMonState.indexOf(']', b1);
            if (b1 >= 0 && b2 > b1) {
                try { j = Integer.parseInt(agentMonState.substring(b1 + 1, b2)); }
                catch (NumberFormatException ignored) {}
            }
        }

        final int target = j;
        ExtendedTransition best = minByKey(ctrl, "purchase[",
            t -> { int i = t.extractFirstIndex(); int idx = 3 + 2 * i; return idx < parts.length && !"Success".equals(parts[idx]); },
            t -> Math.abs(t.extractFirstIndex() - target));
        return best != null ? best : ctrl.get(0);
    }

    /**
     * AT: prefer approach[i] if present; otherwise pick descend[i][j] where:
     *   1. plane i is currently at height j+1 (one step above target — no skipping)
     *   2. height j is Empty in the from-state
     * Among valid candidates, pick the one with lowest j.
     */
    private ExtendedTransition atPickControllable(List<ExtendedTransition> ctrl, boolean canReturnNull) {
        if (canReturnNull) return null;
        for (ExtendedTransition t : ctrl) {
            if (t.action().startsWith("approach[") && heightsAreConsecutive(t.from())) return t;
        }
        ExtendedTransition best = minByKey(ctrl, "descend[",
            t -> { int j = descendY(t.action()); String[] p = t.from().split("\\|"); return p.length > j + 2 && "Empty".equals(p[j + 2]); },
            t -> descendY(t.action()));
        return best != null ? best : ctrl.get(0);
    }

    // ── search helpers ────────────────────────────────────────────────────────

    private ExtendedTransition minByKey(List<ExtendedTransition> ctrl, String prefix,
                                        Predicate<ExtendedTransition> cond, ToIntFunction<ExtendedTransition> keyFn) {
        int best = Integer.MAX_VALUE;
        ExtendedTransition result = null;
        for (ExtendedTransition t : ctrl) {
            if (!t.action().startsWith(prefix) || !cond.test(t)) continue;
            int key = keyFn.applyAsInt(t);
            if (key < best) { best = key; result = t; }
        }
        return result;
    }

    private ExtendedTransition maxByKey(List<ExtendedTransition> ctrl, String prefix,
                                        Predicate<ExtendedTransition> cond, ToIntFunction<ExtendedTransition> keyFn) {
        int best = Integer.MIN_VALUE;
        ExtendedTransition result = null;
        for (ExtendedTransition t : ctrl) {
            if (!t.action().startsWith(prefix) || !cond.test(t)) continue;
            int key = keyFn.applyAsInt(t);
            if (key > best) { best = key; result = t; }
        }
        return result;
    }

    private ExtendedTransition minByComparator(List<ExtendedTransition> ctrl, String prefix,
                                               Comparator<ExtendedTransition> cmp) {
        ExtendedTransition best = null;
        for (ExtendedTransition t : ctrl) {
            if (!t.action().startsWith(prefix)) continue;
            if (best == null || cmp.compare(t, best) < 0) best = t;
        }
        return best;
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

    private int descendY(String action) {
        int b2 = action.indexOf(']');
        int b3 = action.indexOf('[', b2);
        int b4 = action.indexOf(']', b3);
        return Integer.parseInt(action.substring(b3 + 1, b4));
    }
}
