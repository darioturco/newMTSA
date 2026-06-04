package newmtsa.synthesis.heuristics;

import newmtsa.parser.ast.LTS;
import newmtsa.synthesis.Director;
import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.features.FeaturesContext;
import newmtsa.synthesis.features.SuperCustomFeatures;

import java.util.*;
import java.util.Comparator;
import java.util.function.Predicate;
import java.util.function.ToIntFunction;

/**
 * SuperDFS heuristic: depth-first search guided by a stack (pile) and an ignore list.
 * Extends SuperHeuristic, implementing pickControllable() with family-specific logic.
 */
public class SuperDFSHeuristic extends SuperHeuristic {

    // Proof-of-concept: when true, pickControllable() uses only SuperCustomFeatures to
    // select the best controllable transition, bypassing the hand-coded heuristic logic.
    // This is intended solely to verify that the SuperCustomFeatures information is
    // sufficient to replicate the oracle decision without access to the raw state.
    public static boolean USE_CUSTOM_FEATURES = true;

    private final String family;
    private final int    n;
    private final int    k;
    private final int    safePlace;

    private SuperCustomFeatures customFeatures;

    private record ChoiceRecord(int step, ExtendedTransition chosen, List<ExtendedTransition> skipped) {}
    private final List<ChoiceRecord> choices = new ArrayList<>();
    private final List<Integer> pickControllableDecisions = new ArrayList<>();

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
    public void init(SynthesisContext ctx) {
        super.init(ctx);
        if (!USE_CUSTOM_FEATURES) return;

        Set<String> alphabet = new LinkedHashSet<>();
        for (LTS lts : ctx.components()) alphabet.addAll(lts.actions());

        Set<String> goals    = ctx.goals();
        Set<String> errors   = ctx.errors();
        Set<String> explored = ctx.exploredStates();
        Set<String> noneView = new AbstractSet<>() {
            @Override public boolean contains(Object o) {
                return explored.contains(o) && !goals.contains(o) && !errors.contains(o);
            }
            @Override public Iterator<String> iterator() { return Collections.emptyIterator(); }
            @Override public int size() { return 0; }
        };
        Map<String, List<ExtendedTransition>> succMapProxy = new AbstractMap<>() {
            @Override public boolean containsKey(Object key) { return explored.contains(key); }
            @Override public List<ExtendedTransition> get(Object key) { return ctx.successorsOf((String) key); }
            @Override public Set<Map.Entry<String, List<ExtendedTransition>>> entrySet() { return Collections.emptySet(); }
        };

        FeaturesContext featCtx = new FeaturesContext(
                goals, errors, noneView, succMapProxy, new HashMap<>(),
                ctx.controllable(), Collections.unmodifiableSet(alphabet),
                ctx.components(), ctx.componentMarked());
        customFeatures = new SuperCustomFeatures(family, n, k);
        customFeatures.init(featCtx);
    }

    @Override
    protected ExtendedTransition pickControllable(List<ExtendedTransition> ctrl, boolean canReturnNull) {
        subHeuristicCalls++;
        ExtendedTransition chosen = USE_CUSTOM_FEATURES ? pickByFeatures(ctrl, canReturnNull) : switch (family) {
            case "AT" -> atPickControllable(ctrl, canReturnNull);
            case "DP" -> dpPickControllable(ctrl, canReturnNull);
            case "TL" -> tlPickControllable(ctrl, canReturnNull);
            case "BW" -> bwPickControllable(ctrl, canReturnNull);
            case "CM" -> cmPickControllable(ctrl, canReturnNull);
            case "TA" -> taPickControllable(ctrl, canReturnNull);
            default   -> ctrl.get(0);
        };
        if (chosen != null) pickControllableDecisions.add(ctrl.indexOf(chosen));
        return chosen;
    }

    private ExtendedTransition pickByFeatures(List<ExtendedTransition> ctrl, boolean canReturnNull) {
        if (canReturnNull) return null;
        customFeatures.precompute(ctrl);
        float[][] feats = new float[ctrl.size()][];
        for (int i = 0; i < ctrl.size(); i++) feats[i] = customFeatures.compute(ctrl.get(i));
        return switch (family) {
            case "AT" -> pickATByFeatures(ctrl, feats);
            case "DP" -> pickDPByFeatures(ctrl, feats);
            case "BW" -> pickBWByFeatures(ctrl, feats);
            case "CM" -> pickCMByFeatures(ctrl, feats);
            case "TL" -> pickTLByFeatures(ctrl, feats);
            case "TA" -> pickTAByFeatures(ctrl, feats);
            default   -> ctrl.get(0);
        };
    }

    // Features: is_approach[0], target_height_empty[1], heights_consecutive[2], height_idx_norm[3], height_matches_approach_slot[4]
    private ExtendedTransition pickATByFeatures(List<ExtendedTransition> ctrl, float[][] feats) {
        // Priority 1: approach + consecutive heights
        for (int i = 0; i < ctrl.size(); i++)
            if (feats[i][0] == 1f && feats[i][2] == 1f) return ctrl.get(i);
        // Priority 2: target empty + matches waiting-slot assignment
        for (int i = 0; i < ctrl.size(); i++)
            if (feats[i][1] == 1f && feats[i][4] == 1f) return ctrl.get(i);
        // Priority 3: target empty + min height (fallback)
        int bestIdx = -1; float bestHeight = Float.MAX_VALUE;
        for (int i = 0; i < ctrl.size(); i++)
            if (feats[i][1] == 1f && feats[i][3] < bestHeight) { bestHeight = feats[i][3]; bestIdx = i; }
        return bestIdx >= 0 ? ctrl.get(bestIdx) : ctrl.get(0);
    }

    // Features: is_take[0], phil_is_ready[1], phil_is_hungry[2],
    //           is_min_ready_idx[3], is_min_hungry_idx[4], num_ready_norm[5], num_hungry_norm[6]
    private ExtendedTransition pickDPByFeatures(List<ExtendedTransition> ctrl, float[][] feats) {
        for (int i = 0; i < ctrl.size(); i++)
            if (feats[i][0] == 1f && feats[i][1] == 1f && feats[i][3] == 1f) return ctrl.get(i);
        for (int i = 0; i < ctrl.size(); i++)
            if (feats[i][0] == 1f && feats[i][2] == 1f && feats[i][4] == 1f) return ctrl.get(i);
        return ctrl.get(0);
    }

    // Features: is_approve[0], is_refuse[1], is_assign[2], doc_is_rejected[3],
    //           crew_is_pending[4], crew_is_rejected[5], is_min_eligible_assign[6], num_pending_norm[7]
    private ExtendedTransition pickBWByFeatures(List<ExtendedTransition> ctrl, float[][] feats) {
        for (int i = 0; i < ctrl.size(); i++)
            if (feats[i][0] == 1f) return ctrl.get(i);
        for (int i = 0; i < ctrl.size(); i++)
            if (feats[i][1] == 1f && feats[i][3] == 1f) return ctrl.get(i);
        for (int i = 0; i < ctrl.size(); i++)
            if (feats[i][2] == 1f && (feats[i][4] == 1f || feats[i][5] == 1f) && feats[i][6] == 1f) return ctrl.get(i);
        return ctrl.get(0);
    }

    // Features: dist_to_safe_norm[0], is_min_dist_candidate[1], is_min_mouse_among_min_dist[2], current_pos_norm[3]
    private ExtendedTransition pickCMByFeatures(List<ExtendedTransition> ctrl, float[][] feats) {
        for (int i = 0; i < ctrl.size(); i++)
            if (feats[i][1] == 1f && feats[i][2] == 1f) return ctrl.get(i);
        return ctrl.get(0);
    }

    // Features: dest_is_explored[0], is_max_index[1]
    private ExtendedTransition pickTLByFeatures(List<ExtendedTransition> ctrl, float[][] feats) {
        for (int i = 0; i < ctrl.size(); i++)
            if (feats[i][0] == 1f) return ctrl.get(i);
        for (int i = 0; i < ctrl.size(); i++)
            if (feats[i][1] == 1f) return ctrl.get(i);
        return ctrl.get(0);
    }

    // Features: is_agency_succ[0], is_agency_fail[1], is_purchase[2], all_monitors_success[3],
    //           this_monitor_success[4], purchase_to_disallow_dist_norm[5]
    private ExtendedTransition pickTAByFeatures(List<ExtendedTransition> ctrl, float[][] feats) {
        for (int i = 0; i < ctrl.size(); i++)
            if (feats[i][0] == 1f && feats[i][3] == 1f) return ctrl.get(i);
        for (int i = 0; i < ctrl.size(); i++)
            if (feats[i][1] == 1f) return ctrl.get(i);
        // pick purchase where monitor not success, min dist to disallow
        int bestIdx = -1; float bestDist = Float.MAX_VALUE;
        for (int i = 0; i < ctrl.size(); i++)
            if (feats[i][2] == 1f && feats[i][4] == 0f && feats[i][5] < bestDist) { bestDist = feats[i][5]; bestIdx = i; }
        return bestIdx >= 0 ? ctrl.get(bestIdx) : ctrl.get(0);
    }

    @Override
    protected void onControllableChosen(ExtendedTransition chosen, List<ExtendedTransition> others) {
        choices.add(new ChoiceRecord(stepCount, chosen, others));
    }

    @Override
    public void notifyExplorationEnd(Director result) {
        if(ctx.verbose()) {
            System.out.println("[SuperDFS] Sub-heuristic calls: " + subHeuristicCalls);
            System.out.println("[SuperDFS] pickControllable decisions: " + pickControllableDecisions);
        }
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
