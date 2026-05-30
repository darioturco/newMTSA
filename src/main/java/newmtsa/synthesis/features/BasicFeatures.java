package newmtsa.synthesis.features;

import newmtsa.synthesis.ExtendedTransition;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Basic feature set for RL-guided DCS exploration, as described in
 * "On the fly controller synthesis via reinforcement learning".
 *
 * <p>Each frontier transition (s, ℓ, s′) is represented by the concatenation of:
 * <ol>
 *   <li><b>eventLabel</b> ({@code |Σ̃|} bits) — one-hot of ℓ over the no-index alphabet Σ̃.</li>
 *   <li><b>stateLabel</b> ({@code |Σ̃|} bits) — for each action type, whether a
 *       transition of that type arrives at s in the explored graph.</li>
 *   <li><b>controllable</b> (1 bit) — whether ℓ is controllable.</li>
 *   <li><b>markedState</b> (2 bits) — [isMarked(s), isMarked(s′)].</li>
 *   <li><b>currentPhase</b> (3 bits) — [goals found, marked state found,
 *       winning loop closed].</li>
 *   <li><b>childNodeState</b> (3 bits) — [s′∈Goals, s′∈Errors, s′∈None];
 *       all zero means s′ is not yet explored.</li>
 *   <li><b>uncontrollableNeighborhood</b> (4 bits) — uncontrollable transitions
 *       from s and s′ (unexplored / total).</li>
 *   <li><b>exploredStateChild</b> (2 bits) — [s explored, s′ explored].</li>
 *   <li><b>isLastExpanded</b> (2 bits) — [s == lastExpandedTo, s == lastExpandedFrom].</li>
 *   <li><b>childDeadlock</b> (1 bit) — s′ explored and has no outgoing transitions.</li>
 *   <li><b>hasIndex</b> (1 bit) — ℓ contains a digit character.</li>
 * </ol>
 * Total: {@code 2·|Σ̃| + 19} bits.
 */
public class BasicFeatures implements FeatureCompute {

    private FeaturesContext ctx;

    @Override
    public void init(FeaturesContext context) {
        this.ctx = context;
        ctx.getNoIdxAlphabet(); // pre-compute alphabet structures
    }

    @Override
    public float[] compute(ExtendedTransition t) {
        int n     = ctx.getNoIdxAlphabet().size();
        float[] f = new float[2 * n + 19];
        int   p = 0;

        // 1. event label — one-hot of ℓ in Σ̃
        Integer ei = ctx.getNoIdxIndex().get(FeaturesContext.removeIndices(t.action()));
        if (ei != null) f[p + ei] = 1;
        p += n;

        // 2. state label — which action types have transitions arriving at s
        String s = t.from();
        for (String parent : ctx.parents.getOrDefault(s, Set.of())) {
            for (ExtendedTransition pt : ctx.succMap.getOrDefault(parent, List.of())) {
                if (pt.to().equals(s)) {
                    Integer pi = ctx.getNoIdxIndex().get(FeaturesContext.removeIndices(pt.action()));
                    if (pi != null) f[p + pi] = 1;
                }
            }
        }
        p += n;

        // 3. controllable
        f[p++] = ctx.controllable.contains(t.action()) ? 1 : 0;

        // 4. marked state
        f[p++] = ctx.isMarked(t.from()) ? 1 : 0;
        f[p++] = ctx.isMarked(t.to())   ? 1 : 0;

        // 5. current phase
        f[p++] = !ctx.goals.isEmpty()             ? 1 : 0;
        f[p++] = ctx.markedStateFound              ? 1 : 0;
        f[p++] = ctx.closedWinningLoopsCount > 0   ? 1 : 0;

        // 6. child node state
        String sʹ = t.to();
        f[p++] = ctx.goals.contains(sʹ)  ? 1 : 0;
        f[p++] = ctx.errors.contains(sʹ) ? 1 : 0;
        f[p++] = ctx.none.contains(sʹ)   ? 1 : 0;

        // 7. uncontrollable neighborhood
        int sUncTot = 0, sUncUnexp = 0;
        for (ExtendedTransition st : ctx.succMap.getOrDefault(s, List.of())) {
            if (!ctx.controllable.contains(st.action())) {
                sUncTot++;
                if (!ctx.isVisited(st.to())) sUncUnexp++;
            }
        }
        List<ExtendedTransition> sʹSucc = ctx.succMap.get(sʹ);
        int sʹUncTot, sʹUncUnexp;
        if (sʹSucc == null) {
            // s' not yet explored — conservatively assume it has uncontrollable transitions
            sʹUncTot = 1; sʹUncUnexp = 1;
        } else {
            sʹUncTot = 0; sʹUncUnexp = 0;
            for (ExtendedTransition st : sʹSucc) {
                if (!ctx.controllable.contains(st.action())) {
                    sʹUncTot++;
                    if (!ctx.isVisited(st.to())) sʹUncUnexp++;
                }
            }
        }
        f[p++] = sUncUnexp  > 0 ? 1 : 0;
        f[p++] = sUncTot    > 0 ? 1 : 0;
        f[p++] = sʹUncUnexp > 0 ? 1 : 0;
        f[p++] = sʹUncTot   > 0 ? 1 : 0;

        // 8. explored state child
        f[p++] = ctx.succMap.containsKey(s)  ? 1 : 0;   // always 1 for frontier transitions
        f[p++] = ctx.succMap.containsKey(sʹ) ? 1 : 0;

        // 9. is last expanded
        f[p++] = s.equals(ctx.lastExpandedTo)   ? 1 : 0;
        f[p++] = s.equals(ctx.lastExpandedFrom) ? 1 : 0;

        // 10. child deadlock
        f[p++] = (sʹSucc != null && sʹSucc.isEmpty()) ? 1 : 0;

        // 11. has index
        boolean hasIdx = false;
        for (char c : t.action().toCharArray()) {
            if (Character.isDigit(c)) { hasIdx = true; break; }
        }
        f[p] = hasIdx ? 1 : 0;

        return f;
    }

    @Override
    public List<String> getFeatureNames() {
        List<String> names = new ArrayList<>();
        List<String> alphabet = ctx.getNoIdxAlphabet();
        for (String a : alphabet) names.add("event_" + a);
        for (String a : alphabet) names.add("state_" + a);
        names.add("controllable");
        names.add("marked_from");
        names.add("marked_to");
        names.add("goals_found");
        names.add("marked_found");
        names.add("loop_closed");
        names.add("child_goal");
        names.add("child_error");
        names.add("child_none");
        names.add("src_unc_unexp");
        names.add("src_unc_total");
        names.add("child_unc_unexp");
        names.add("child_unc_total");
        names.add("src_explored");
        names.add("child_explored");
        names.add("is_last_to");
        names.add("is_last_from");
        names.add("child_deadlock");
        names.add("has_index");
        return names;
    }

    @Override
    public String getFeatureGroupName() {
        return "BASIC";
    }
}
