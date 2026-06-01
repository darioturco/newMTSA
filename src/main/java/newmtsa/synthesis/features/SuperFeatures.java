package newmtsa.synthesis.features;

import newmtsa.synthesis.ExtendedTransition;

import java.util.*;

/**
 * Super feature set for RL-guided DCS exploration.
 *
 * <p>Designed for a per-transition scoring architecture: each candidate controllable
 * transition is scored independently by the same model, and the highest-scoring
 * transition is selected.  All features are size-normalised so that a model trained
 * on small instances generalises to larger ones.
 *
 * <p>Feature vector layout (total = {@code |Σ̃| + 3·|R̃| + 11}):
 * <ol>
 *   <li><b>abstract_action</b> ({@code |Σ̃|}) — one-hot of the abstract action (indices stripped).</li>
 *   <li><b>first_index_norm</b> (1) — first bracketed index scaled to [0,1] as
 *       {@code i/numComponents}; 0 if the action carries no index.</li>
 *   <li><b>second_index_norm</b> (1) — second bracketed index, same scaling; 0 if absent.</li>
 *   <li><b>dest_explored</b> (1) — 1 if the target state has been expanded.</li>
 *   <li><b>dest_goal</b> (1) — 1 if the target state is a goal.</li>
 *   <li><b>dest_error</b> (1) — 1 if the target state is an error.</li>
 *   <li><b>dest_none</b> (1) — 1 if the target state is classified as none.</li>
 *   <li><b>dest_unc_unexp</b> (1) — 1 if the target has any unexplored uncontrollable successor
 *       (conservatively 1 when the target is not yet expanded).</li>
 *   <li><b>dest_unc_total</b> (1) — 1 if the target has any uncontrollable successor at all.</li>
 *   <li><b>role_frac</b> ({@code |R̃|}) — for each abstract role: fraction of components in
 *       the source state that have that role, in [0,1].</li>
 *   <li><b>changed_from</b> ({@code |R̃|}) — binary: which abstract roles appear as the
 *       <em>source</em> role of any component that changes role on this transition.</li>
 *   <li><b>changed_to</b> ({@code |R̃|}) — binary: which abstract roles appear as the
 *       <em>target</em> role of any component that changes role on this transition.</li>
 *   <li><b>phase_goals_found</b> (1) — 1 if the goals set is non-empty.</li>
 *   <li><b>phase_marked_found</b> (1) — 1 if a marked state has been encountered.</li>
 *   <li><b>phase_loop_closed</b> (1) — 1 if at least one winning loop has been closed.</li>
 * </ol>
 */
public class SuperFeatures implements FeatureCompute {

    private FeaturesContext ctx;
    private List<String>    abstractRoles;
    private int             numComponents;

    @Override
    public void init(FeaturesContext context) {
        this.ctx           = context;
        this.numComponents = context.getComponents().size();

        Set<String> allRoles = new TreeSet<>();
        for (var lts : context.getComponents()) {
            for (String state : lts.states()) {
                allRoles.add(abstractRole(state));
            }
        }
        this.abstractRoles = new ArrayList<>(allRoles);

        ctx.getNoIdxAlphabet();
    }

    @Override
    public float[] compute(ExtendedTransition t) {
        int n   = ctx.getNoIdxAlphabet().size();
        int r   = abstractRoles.size();
        float[] f = new float[n + 3 * r + 11];
        int     p = 0;

        // 1. abstract_action one-hot
        Integer ai = ctx.getNoIdxIndex().get(FeaturesContext.removeIndices(t.action()));
        if (ai != null) f[p + ai] = 1;
        p += n;

        // 2. first_index_norm
        f[p++] = firstIndexNorm(t.action());
        // 3. second_index_norm
        f[p++] = secondIndexNorm(t.action());

        // 4-9. destination state features
        String dest = t.to();
        f[p++] = ctx.succMap.containsKey(dest) ? 1 : 0;
        f[p++] = ctx.goals.contains(dest)       ? 1 : 0;
        f[p++] = ctx.errors.contains(dest)      ? 1 : 0;
        f[p++] = ctx.none.contains(dest)        ? 1 : 0;

        List<ExtendedTransition> destSucc = ctx.succMap.get(dest);
        if (destSucc == null) {
            f[p++] = 1; // dest_unc_unexp — unexplored, assume uncontrollable exists
            f[p++] = 1; // dest_unc_total
        } else {
            int uncUnexp = 0, uncTotal = 0;
            for (ExtendedTransition s : destSucc) {
                if (!ctx.controllable.contains(s.action())) {
                    uncTotal++;
                    if (!ctx.isVisited(s.to())) uncUnexp++;
                }
            }
            f[p++] = uncUnexp > 0 ? 1 : 0;
            f[p++] = uncTotal > 0 ? 1 : 0;
        }

        // 10. role_frac: fraction of components in source state with each abstract role
        String[] fromParts = splitAndClean(t.from());
        String[] toParts   = splitAndClean(t.to());

        for (String part : fromParts) {
            int idx = Collections.binarySearch(abstractRoles, abstractRole(part));
            if (idx >= 0) f[p + idx] += 1.0f / numComponents;
        }
        p += r;

        // 11-12. changed_from_role and changed_to_role
        int changedFromBase = p;
        int changedToBase   = p + r;
        int len = Math.min(fromParts.length, toParts.length);
        for (int i = 0; i < len; i++) {
            String fromRole = abstractRole(fromParts[i]);
            String toRole   = abstractRole(toParts[i]);
            if (!fromRole.equals(toRole)) {
                int fi = Collections.binarySearch(abstractRoles, fromRole);
                int ti = Collections.binarySearch(abstractRoles, toRole);
                if (fi >= 0) f[changedFromBase + fi] = 1;
                if (ti >= 0) f[changedToBase   + ti] = 1;
            }
        }
        p += 2 * r;

        // 13-15. phase
        f[p++] = !ctx.goals.isEmpty()           ? 1 : 0;
        f[p++] = ctx.markedStateFound            ? 1 : 0;
        f[p]   = ctx.closedWinningLoopsCount > 0 ? 1 : 0;

        return f;
    }

    @Override
    public List<String> getFeatureNames() {
        List<String> names = new ArrayList<>();
        for (String a : ctx.getNoIdxAlphabet()) names.add("action_" + a);
        names.add("first_index_norm");
        names.add("second_index_norm");
        names.add("dest_explored");
        names.add("dest_goal");
        names.add("dest_error");
        names.add("dest_none");
        names.add("dest_unc_unexp");
        names.add("dest_unc_total");
        for (String role : abstractRoles) names.add("role_frac_" + role);
        for (String role : abstractRoles) names.add("changed_from_" + role);
        for (String role : abstractRoles) names.add("changed_to_"   + role);
        names.add("phase_goals_found");
        names.add("phase_marked_found");
        names.add("phase_loop_closed");
        return names;
    }

    @Override
    public String getFeatureGroupName() {
        return "SUPER";
    }

    private String[] splitAndClean(String composite) {
        String[] parts = composite.split("\\|", numComponents);
        for (int i = 0; i < parts.length; i++) {
            int hash = parts[i].indexOf('#');
            if (hash >= 0) parts[i] = parts[i].substring(0, hash);
        }
        return parts;
    }

    private static String abstractRole(String state) {
        String s = FeaturesContext.removeIndices(state);   // strips [N]
        s = s.replaceAll("_[0-9]+", "");                   // strips _N (FSP split-state suffixes)
        s = s.replaceAll("[0-9]+$", "");                   // strips trailing bare digits
        return s.replaceAll("\\(\\d+(?:,\\s*\\d+)*\\)$", "");
    }

    private float firstIndexNorm(String action) {
        int b1 = action.indexOf('[');
        if (b1 < 0) return 0;
        int b2 = action.indexOf(']', b1 + 1);
        if (b2 < 0) return 0;
        try {
            return Integer.parseInt(action.substring(b1 + 1, b2)) / (float) numComponents;
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    private float secondIndexNorm(String action) {
        int b1 = action.indexOf('[');
        if (b1 < 0) return 0;
        int b2 = action.indexOf('[', b1 + 1);
        if (b2 < 0) return 0;
        int b3 = action.indexOf(']', b2 + 1);
        if (b3 < 0) return 0;
        try {
            return Integer.parseInt(action.substring(b2 + 1, b3)) / (float) numComponents;
        } catch (NumberFormatException e) {
            return 0;
        }
    }
}
