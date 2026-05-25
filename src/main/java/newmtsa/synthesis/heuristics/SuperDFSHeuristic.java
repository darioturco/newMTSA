package newmtsa.synthesis.heuristics;

import newmtsa.synthesis.Director;
import newmtsa.synthesis.ExtendedTransition;

import java.util.*;
import java.util.Comparator;
import java.util.function.Predicate;
import java.util.function.ToIntFunction;

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
            boolean mixed = !ctrl.isEmpty() && !decidedStates.contains(currentState);
            if (mixed) {
                // Case 5: both noncontrollable and controllable present.
                // Ask the family picker whether to choose a controllable (canReturnNull=true).
                // null => expand noncontrollable as usual.
                // non-null => pick that controllable, push all noncontrollables to stack,
                //             push remaining controllables to ignore.
                ExtendedTransition ctrlChoice = pickControllable(ctrl, true);
                if (ctrlChoice != null) {
                    chosen = ctrlChoice;
                    for (int i = nonCtrl.size() - 1; i >= 0; i--) {
                        stack.push(nonCtrl.get(i));
                    }
                    List<ExtendedTransition> others = new ArrayList<>();
                    for (ExtendedTransition t : ctrl) {
                        if (t != ctrlChoice) others.add(t);
                    }
                    ignored.addAll(others);
                    choices.add(new ChoiceRecord(stepCount, chosen, others));
                    decidedStates.add(currentState);
                } else {
                    chosen = nonCtrl.get(0);
                    for (int i = nonCtrl.size() - 1; i >= 1; i--) {
                        stack.push(nonCtrl.get(i));
                    }
                }
            } else {
                chosen = nonCtrl.get(0);
                for (int i = nonCtrl.size() - 1; i >= 1; i--) {
                    stack.push(nonCtrl.get(i));
                }
            }
        } else if (!ctrl.isEmpty()) {
            if (decidedStates.contains(currentState)) {
                return pickFromStackOrIgnored(pending);
            }
            chosen = pickControllable(ctrl, false);
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

    /**
     * @param canReturnNull true only in Case 5 (mixed nonctrl+ctrl).
     *                      Return null to signal "don't pick a controllable; expand noncontrollable instead."
     */
    private ExtendedTransition pickControllable(List<ExtendedTransition> ctrl, boolean canReturnNull) {
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

    // ── backtrack ─────────────────────────────────────────────────────────────

    @Override
    public void notifyExplorationEnd(Director result) {

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
