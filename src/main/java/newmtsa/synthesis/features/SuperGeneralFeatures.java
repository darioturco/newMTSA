package newmtsa.synthesis.features;

import newmtsa.synthesis.ExtendedTransition;

import java.util.*;

/**
 * General-purpose feature set for RL-guided DCS exploration.
 *
 * <p>Designed as a family-agnostic replacement for {@link SuperCustomFeatures}: it aims for
 * the same decision power without hardcoding any family-specific substate string (no
 * {@code "Empty"}, {@code "Ready"}, {@code "Success"}, …) and without per-family code paths.
 * Like {@link SuperFeatures} it uses a per-transition scoring architecture — each candidate
 * controllable transition is scored independently and the highest-scoring one is selected —
 * and every feature is size-normalised so a model trained on small instances generalises to
 * larger ones.
 *
 * <p>The key capability over {@link SuperFeatures} is <b>cross-candidate index ranking</b>,
 * computed in {@link #precompute(List)}: the oracle decisions that {@code SuperCustomFeatures}
 * encodes by hand ("min eligible index", "max get index", "closest to safe position") all
 * reduce to ordering candidates of the same abstract action by their bracketed index. Those
 * ranks are exposed here generically so the agent can learn the same policy.
 *
 * <p>Feature vector layout — full (BW/TL/CM): {@code |Σ̃c| + 3·|R̃| + 27}; slim (DP/AT/TA): {@code |Σ̃c| + 2·|R̃| + 12}:
 * <ol>
 *   <li><b>abstract_action</b> ({@code |Σ̃c|}) — one-hot of the abstract action (indices stripped),
 *       over the <em>controllable</em> alphabet only (SuperRL scores controllable candidates).</li>
 *   <li><b>first_index_norm</b>, <b>second_index_norm</b> (2) — bracketed indices scaled by
 *       {@code numComponents}; 0 when absent.</li>
 *   <li><b>cross-candidate index ranking</b> (9) — see {@link #precompute(List)}.</li>
 *   <li><b>destination / explored-graph structure</b> (11) + <b>changed_substate_explored</b> (1).</li>
 *   <li><b>role_frac</b> ({@code |R̃|}), <b>changed_from</b> ({@code |R̃|}),
 *       <b>changed_to</b> ({@code |R̃|}), <b>num_changed_components_norm</b> (1).</li>
 *   <li><b>phase</b> (3).</li>
 * </ol>
 *
 * <p>Vector length is invariant across instances of the same family with {@code n >= 2},
 * assuming a shared no-index alphabet and shared LTS component templates: {@code |Σ̃c|} and
 * {@code |R̃|} derive from the templates, and every other group is fixed-width.
 */
public class SuperGeneralFeatures implements FeatureCompute {

    private FeaturesContext ctx;
    private List<String>    abstractRoles;
    private int             numComponents;

    // controllable-only no-index alphabet (SuperRL scores controllable candidates only)
    private List<String>         ctrlAlphabet;
    private Map<String, Integer> ctrlIndex;

    /**
     * When {@code true}, drops all floating-point (non-binary) features plus the
     * explored-graph / phase features:
     * removed floats: first_index_norm, second_index_norm, first_idx_rank_in_action_norm,
     *   dist_to_center_idx_norm, role_frac_* (|R̃| features), num_changed_components_norm.
     * Slim (DP/AT/TA) vector size: {@code |Σ̃c| + 2·|R̃| + 12}.  Full (others incl. CM): {@code |Σ̃c| + 3·|R̃| + 27}.
     * The 10 graph/phase features (dest_error, dest_none, dest_deadlock, dest_unc_*, src_unc_*,
     * phase_*) are permanently removed for all families.
     * CM uses full mode so that second_index_norm (board position) and role_frac_* (served-mice
     * context) are available to resolve oracle-tie inconsistencies.
     * Set at {@link #init} time from the {@code superdfs.family} system property.
     */
    private boolean slimMode;

    // ── cross-candidate caches (rebuilt each precompute) ──────────────────────
    // abstract action → {minFirstIdx, maxFirstIdx}
    private final Map<String, int[]> firstIdxRange = new HashMap<>();
    // abstract action → {minLastIdx, maxLastIdx}
    private final Map<String, int[]> lastIdxRange  = new HashMap<>();
    // abstract action → sorted distinct first indices (for rank)
    private final Map<String, int[]> firstIdxSorted = new HashMap<>();
    // abstract action → sorted distinct last indices (for last-index center distance)
    private final Map<String, int[]> lastIdxSorted  = new HashMap<>();
    private int globalMinFirstIdx = Integer.MAX_VALUE;
    private int globalMaxFirstIdx = Integer.MIN_VALUE;

    // Per-position set of component substates seen in expanded states (cumulative across the
    // search). A from-state of any decision is, by definition, an explored/expanded state, so
    // recording from-states builds the set of (position, substate) pairs witnessed so far.
    // Works in both training and ONNX-deploy modes (does not need to iterate succMap).
    private List<Set<String>> seenSubstateByPos;

    @Override
    public void init(FeaturesContext context) {
        this.ctx           = context;
        this.numComponents = context.getComponents().size();
        String _fam = System.getProperty("superdfs.family", "");
        // CM excluded: needs second_index_norm (board position) + role_frac (served-mice context)
        // to distinguish candidates that share binary ranking flags but differ in actual position.
        this.slimMode = "DP".equals(_fam) || "AT".equals(_fam) || "TA".equals(_fam);

        Set<String> allRoles = new TreeSet<>();
        for (var lts : context.getComponents()) {
            for (String state : lts.states()) {
                allRoles.add(abstractRole(state));
            }
        }
        this.abstractRoles = new ArrayList<>(allRoles);

        // controllable no-index alphabet: strip [N] from each controllable action, dedupe, sort
        TreeSet<String> ctrl = new TreeSet<>();
        for (String a : context.controllable) ctrl.add(FeaturesContext.removeIndices(a));
        this.ctrlAlphabet = new ArrayList<>(ctrl);
        this.ctrlIndex    = new HashMap<>();
        for (int i = 0; i < ctrlAlphabet.size(); i++) ctrlIndex.put(ctrlAlphabet.get(i), i);

        seenSubstateByPos = new ArrayList<>(numComponents);
        for (int i = 0; i < numComponents; i++) seenSubstateByPos.add(new HashSet<>());
    }

    @Override
    public void precompute(List<ExtendedTransition> candidates) {
        firstIdxRange.clear();
        lastIdxRange.clear();
        firstIdxSorted.clear();
        lastIdxSorted.clear();
        globalMinFirstIdx = Integer.MAX_VALUE;
        globalMaxFirstIdx = Integer.MIN_VALUE;

        Map<String, TreeSet<Integer>> firstByAction = new HashMap<>();
        Map<String, TreeSet<Integer>> lastByAction  = new HashMap<>();
        for (ExtendedTransition t : candidates) {
            String aa  = FeaturesContext.removeIndices(t.action());
            int    fst = firstIndex(t.action());
            if (fst != NO_INDEX) {
                int[] fr = firstIdxRange.computeIfAbsent(aa, k -> new int[]{Integer.MAX_VALUE, Integer.MIN_VALUE});
                if (fst < fr[0]) fr[0] = fst;
                if (fst > fr[1]) fr[1] = fst;
                firstByAction.computeIfAbsent(aa, k -> new TreeSet<>()).add(fst);
                if (fst < globalMinFirstIdx) globalMinFirstIdx = fst;
                if (fst > globalMaxFirstIdx) globalMaxFirstIdx = fst;
            }
            int lst = lastIndex(t.action());
            if (lst != NO_INDEX) {
                int[] lr = lastIdxRange.computeIfAbsent(aa, k -> new int[]{Integer.MAX_VALUE, Integer.MIN_VALUE});
                if (lst < lr[0]) lr[0] = lst;
                if (lst > lr[1]) lr[1] = lst;
                lastByAction.computeIfAbsent(aa, k -> new TreeSet<>()).add(lst);
            }
        }
        for (var e : firstByAction.entrySet()) {
            int[] arr = new int[e.getValue().size()];
            int i = 0;
            for (int v : e.getValue()) arr[i++] = v;
            firstIdxSorted.put(e.getKey(), arr);
        }
        for (var e : lastByAction.entrySet()) {
            int[] arr = new int[e.getValue().size()];
            int i = 0;
            for (int v : e.getValue()) arr[i++] = v;
            lastIdxSorted.put(e.getKey(), arr);
        }

        // Record the decision's from-state (an explored/expanded state) into the per-position
        // seen-substate sets. All candidates share the same from-state.
        if (!candidates.isEmpty()) {
            String[] fromParts = splitAndClean(candidates.get(0).from());
            for (int i = 0; i < fromParts.length && i < seenSubstateByPos.size(); i++) {
                seenSubstateByPos.get(i).add(fromParts[i]);
            }
        }
    }

    @Override
    public float[] compute(ExtendedTransition t) {
        int n = ctrlAlphabet.size();
        int r = abstractRoles.size();
        float[] f = new float[slimMode ? n + 2 * r + 12 : n + 3 * r + 27];
        int     p = 0;

        String action = t.action();
        String aa     = FeaturesContext.removeIndices(action);

        // ── A. action identity ────────────────────────────────────────────────
        Integer ai = ctrlIndex.get(aa);
        if (ai != null) f[p + ai] = 1;
        p += n;

        if (!slimMode) {
            f[p++] = firstIndexNorm(action);
            f[p++] = secondIndexNorm(action);
        }

        // ── B. cross-candidate index ranking ──────────────────────────────────
        int   fst       = firstIndex(action);
        int   lst       = lastIndex(action);
        int[] fr        = firstIdxRange.get(aa);
        int[] lr        = lastIdxRange.get(aa);
        int[] sorted    = firstIdxSorted.get(aa);

        boolean isMinFirstInAction = fst != NO_INDEX && fr != null && fst == fr[0];
        boolean isMaxFirstInAction = fst != NO_INDEX && fr != null && fst == fr[1];
        boolean isMinLastInAction  = lst != NO_INDEX && lr != null && lst == lr[0];
        boolean isMaxLastInAction  = lst != NO_INDEX && lr != null && lst == lr[1];

        float rankNorm = 0f;
        if (fst != NO_INDEX && sorted != null && sorted.length > 1) {
            int rank = Arrays.binarySearch(sorted, fst);
            if (rank >= 0) rankNorm = rank / (float) (sorted.length - 1);
        }

        // distance to centre of the action's first-index range
        float distToCenterNorm = 0f;
        if (fst != NO_INDEX && fr != null) {
            float center    = (fr[0] + fr[1]) / 2.0f;
            float halfRange = (fr[1] - fr[0]) / 2.0f;
            distToCenterNorm = halfRange > 0 ? Math.abs(fst - center) / halfRange : 0f;
        }

        // closest to centre of last-index range (for CM: destination position closest to safe place)
        boolean isClosestCenterLastIdx = false;
        if (lst != NO_INDEX && lr != null) {
            float lastCenter   = (lr[0] + lr[1]) / 2.0f;
            int[] lastSorted   = lastIdxSorted.get(aa);
            float dist         = Math.abs(lst - lastCenter);
            isClosestCenterLastIdx = dist <= minDistToCenter(lastSorted, lastCenter) + 1e-6f;
        }

        boolean isMinFirstGlobal = fst != NO_INDEX && fst == globalMinFirstIdx;
        boolean isMaxFirstGlobal = fst != NO_INDEX && fst == globalMaxFirstIdx;

        f[p++] = isMinFirstInAction    ? 1f : 0f;
        f[p++] = isMaxFirstInAction    ? 1f : 0f;
        if (!slimMode) f[p++] = rankNorm;
        f[p++] = isMinLastInAction     ? 1f : 0f;
        f[p++] = isMaxLastInAction     ? 1f : 0f;
        if (!slimMode) f[p++] = distToCenterNorm;
        f[p++] = isClosestCenterLastIdx ? 1f : 0f;
        f[p++] = isMinFirstGlobal      ? 1f : 0f;
        f[p++] = isMaxFirstGlobal      ? 1f : 0f;

        // ── C. destination / explored-graph structure ─────────────────────────
        String src  = t.from();
        String dest = t.to();
        f[p++] = ctx.succMap.containsKey(dest) ? 1f : 0f;   // dest_explored
        f[p++] = ctx.goals.contains(dest)       ? 1f : 0f;   // dest_goal

        if (!slimMode) {
            f[p++] = ctx.errors.contains(dest)      ? 1f : 0f;   // dest_error
            f[p++] = ctx.none.contains(dest)        ? 1f : 0f;   // dest_none

            List<ExtendedTransition> destSucc = ctx.succMap.get(dest);
            f[p++] = (destSucc != null && destSucc.isEmpty()) ? 1f : 0f; // dest_deadlock
            if (destSucc == null) {
                f[p++] = 1f; // dest_unc_unexp
                f[p++] = 1f; // dest_unc_total
            } else {
                int uncUnexp = 0, uncTotal = 0;
                for (ExtendedTransition s : destSucc) {
                    if (!ctx.controllable.contains(s.action())) {
                        uncTotal++;
                        if (!ctx.isVisited(s.to())) uncUnexp++;
                    }
                }
                f[p++] = uncUnexp > 0 ? 1f : 0f;
                f[p++] = uncTotal > 0 ? 1f : 0f;
            }

            int srcUncUnexp = 0, srcUncTotal = 0;
            for (ExtendedTransition s : ctx.succMap.getOrDefault(src, List.of())) {
                if (!ctx.controllable.contains(s.action())) {
                    srcUncTotal++;
                    if (!ctx.isVisited(s.to())) srcUncUnexp++;
                }
            }
            f[p++] = srcUncUnexp > 0 ? 1f : 0f;
            f[p++] = srcUncTotal > 0 ? 1f : 0f;
        }

        f[p++] = ctx.isMarked(src)  ? 1f : 0f;   // marked_from
        f[p++] = ctx.isMarked(dest) ? 1f : 0f;   // marked_to

        String[] fromParts = splitAndClean(src);
        String[] toParts   = splitAndClean(dest);

        // changed_substate_explored: 1 if any component this transition CHANGES lands on a
        // substate already witnessed at that position in an explored state — i.e. the move
        // re-creates an already-seen local config (e.g. serving an already-satisfied component),
        // rather than producing genuinely new local progress.
        boolean substateSeen = false;
        int cs = Math.min(fromParts.length, toParts.length);
        for (int i = 0; i < cs; i++) {
            if (!fromParts[i].equals(toParts[i])
                && i < seenSubstateByPos.size()
                && seenSubstateByPos.get(i).contains(toParts[i])) {
                substateSeen = true;
                break;
            }
        }
        f[p++] = substateSeen ? 1f : 0f;

        // ── D. role / state-progress ──────────────────────────────────────────
        if (!slimMode) {
            for (String part : fromParts) {
                int idx = Collections.binarySearch(abstractRoles, abstractRole(part));
                if (idx >= 0) f[p + idx] += 1.0f / numComponents;
            }
            p += r;
        }

        int changedFromBase = p;
        int changedToBase   = p + r;
        int len             = Math.min(fromParts.length, toParts.length);
        int numChanged      = 0;
        for (int i = 0; i < len; i++) {
            String fromRole = abstractRole(fromParts[i]);
            String toRole   = abstractRole(toParts[i]);
            if (!fromRole.equals(toRole)) {
                numChanged++;
                int fi = Collections.binarySearch(abstractRoles, fromRole);
                int ti = Collections.binarySearch(abstractRoles, toRole);
                if (fi >= 0) f[changedFromBase + fi] = 1f;
                if (ti >= 0) f[changedToBase   + ti] = 1f;
            }
        }
        p += 2 * r;
        if (!slimMode) f[p++] = numComponents > 0 ? numChanged / (float) numComponents : 0f;

        // ── E. phase ──────────────────────────────────────────────────────────
        if (!slimMode) {
            f[p++] = !ctx.goals.isEmpty()           ? 1f : 0f;
            f[p++] = ctx.markedStateFound            ? 1f : 0f;
            f[p]   = ctx.closedWinningLoopsCount > 0 ? 1f : 0f;
        }

        return f;
    }

    @Override
    public List<String> getFeatureNames() {
        List<String> names = new ArrayList<>();
        for (String a : ctrlAlphabet) names.add("action_" + a);
        if (!slimMode) {
            names.add("first_index_norm");
            names.add("second_index_norm");
        }

        names.add("is_min_first_idx_in_action");
        names.add("is_max_first_idx_in_action");
        if (!slimMode) names.add("first_idx_rank_in_action_norm");
        names.add("is_min_last_idx_in_action");
        names.add("is_max_last_idx_in_action");
        if (!slimMode) names.add("dist_to_center_idx_norm");
        names.add("is_closest_center_last_idx_in_action");
        names.add("is_min_first_idx_global");
        names.add("is_max_first_idx_global");

        names.add("dest_explored");
        names.add("dest_goal");
        if (!slimMode) {
            names.add("dest_error");
            names.add("dest_none");
            names.add("dest_deadlock");
            names.add("dest_unc_unexp");
            names.add("dest_unc_total");
            names.add("src_unc_unexp");
            names.add("src_unc_total");
        }
        names.add("marked_from");
        names.add("marked_to");
        names.add("changed_substate_explored");

        if (!slimMode) for (String role : abstractRoles) names.add("role_frac_" + role);
        for (String role : abstractRoles) names.add("changed_from_" + role);
        for (String role : abstractRoles) names.add("changed_to_"   + role);
        if (!slimMode) names.add("num_changed_components_norm");

        if (!slimMode) {
            names.add("phase_goals_found");
            names.add("phase_marked_found");
            names.add("phase_loop_closed");
        }
        return names;
    }

    @Override
    public String getFeatureGroupName() {
        return "SUPER_GENERAL";
    }

    // ── helpers (shared design with SuperFeatures) ────────────────────────────

    private static final int NO_INDEX = Integer.MIN_VALUE;

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

    /** First bracketed integer of {@code action}, or {@link #NO_INDEX} if none. */
    private static int firstIndex(String action) {
        int b1 = action.indexOf('[');
        if (b1 < 0) return NO_INDEX;
        int b2 = action.indexOf(']', b1 + 1);
        if (b2 < 0) return NO_INDEX;
        try {
            return Integer.parseInt(action.substring(b1 + 1, b2));
        } catch (NumberFormatException e) {
            return NO_INDEX;
        }
    }

    /** Last bracketed integer of {@code action} (e.g. {@code mouse[i][move[j]]} → j), or {@link #NO_INDEX}.
     *  Skips non-integer brackets (e.g. the outer {@code ]} of {@code [move[j]]}) by scanning left. */
    private static int lastIndex(String action) {
        int end = action.length();
        while (end > 0) {
            int b2 = action.lastIndexOf(']', end - 1);
            if (b2 < 0) return NO_INDEX;
            int b1 = action.lastIndexOf('[', b2);
            if (b1 < 0) return NO_INDEX;
            try {
                return Integer.parseInt(action.substring(b1 + 1, b2));
            } catch (NumberFormatException e) {
                end = b2; // outer bracket — try the next ] to the left
            }
        }
        return NO_INDEX;
    }

    private float firstIndexNorm(String action) {
        int v = firstIndex(action);
        return v == NO_INDEX ? 0f : v / (float) numComponents;
    }

    private float secondIndexNorm(String action) {
        int b1 = action.indexOf('[');
        if (b1 < 0) return 0f;
        int b2 = action.indexOf('[', b1 + 1);
        if (b2 < 0) return 0f;
        int b3 = action.indexOf(']', b2 + 1);
        if (b3 < 0) return 0f;
        try {
            return Integer.parseInt(action.substring(b2 + 1, b3)) / (float) numComponents;
        } catch (NumberFormatException e) {
            return 0f;
        }
    }

    private static float minDistToCenter(int[] sorted, float center) {
        if (sorted == null || sorted.length == 0) return 0f;
        float best = Float.MAX_VALUE;
        for (int v : sorted) best = Math.min(best, Math.abs(v - center));
        return best;
    }
}
