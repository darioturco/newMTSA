package newmtsa.synthesis.features;

import newmtsa.synthesis.ExtendedTransition;

import java.util.List;

/**
 * Family-specific feature set for RL-guided DCS exploration.
 *
 * <p>Each family's feature vector encodes exactly the decision-relevant state
 * that {@code SuperDFSHeuristic} uses, enabling a per-family RL network to
 * learn to imitate the oracle heuristic.
 *
 * <p>Feature group name: {@code "SUPER_CUSTOM"}.
 *
 * <p>Feature vector sizes by family:
 * <ul>
 *   <li>AT — 5 features</li>
 *   <li>DP — 5 features</li>
 *   <li>BW — 7 features</li>
 *   <li>CM — 3 features</li>
 *   <li>TL — 2 features</li>
 *   <li>TA — 6 features</li>
 * </ul>
 *
 * <p>{@link #precompute(List)} must be called with the full candidate list before
 * any {@link #compute(ExtendedTransition)} calls for CM and TL families, which
 * require cross-candidate aggregates.
 */
public class SuperCustomFeatures implements FeatureCompute {

    private final String family;
    private final int    n;
    private final int    k;
    private final int    safePlace;

    private FeaturesContext ctx;

    // cross-candidate cache for CM and TL (populated by precompute)
    private int cachedMinDist      = Integer.MAX_VALUE;
    private int cachedMinDistMouse = Integer.MAX_VALUE;
    private int cachedMaxGetIdx    = Integer.MIN_VALUE;

    public SuperCustomFeatures(String family, int n, int k) {
        this.family    = family;
        this.n         = n;
        this.k         = k;
        this.safePlace = (2 * k + 1) / 2;
    }

    @Override
    public void init(FeaturesContext context) {
        this.ctx = context;
    }

    @Override
    public void precompute(List<ExtendedTransition> candidates) {
        switch (family) {
            case "CM" -> {
                cachedMinDist      = Integer.MAX_VALUE;
                cachedMinDistMouse = Integer.MAX_VALUE;
                for (ExtendedTransition t : candidates) {
                    if (!t.action().startsWith("mouse[")) continue;
                    int pos   = t.extractLastIndex();
                    int dist  = Math.abs(pos - safePlace);
                    int mouse = t.extractFirstIndex();
                    if (dist < cachedMinDist || (dist == cachedMinDist && mouse < cachedMinDistMouse)) {
                        cachedMinDist      = dist;
                        cachedMinDistMouse = mouse;
                    }
                }
            }
            case "TL" -> {
                cachedMaxGetIdx = Integer.MIN_VALUE;
                for (ExtendedTransition t : candidates) {
                    if (!t.action().startsWith("get[")) continue;
                    int idx = t.extractFirstIndex();
                    if (idx > cachedMaxGetIdx) cachedMaxGetIdx = idx;
                }
            }
        }
    }

    @Override
    public float[] compute(ExtendedTransition t) {
        return switch (family) {
            case "AT" -> computeAT(t);
            case "DP" -> computeDP(t);
            case "BW" -> computeBW(t);
            case "CM" -> computeCM(t);
            case "TL" -> computeTL(t);
            case "TA" -> computeTA(t);
            default   -> new float[0];
        };
    }

    // ── AT ────────────────────────────────────────────────────────────────────

    /**
     * AT features (5):
     * is_approach, target_height_empty, heights_consecutive, height_idx_norm,
     * height_matches_approach_slot.
     *
     * <p>State layout: parts[0]=ResponseMonitor, parts[1]=RampMonitor,
     * parts[2+h]=height-h slot ("Empty" or "Occupied[p]"),
     * parts[2+k+q]=state of airplane q ("Airplane(q)", "Holding[h]", "End", …).
     *
     * <p>height_matches_approach_slot: for descend[p][h], equals 1 iff h equals the
     * number of lower-indexed planes (q&lt;p) still in "Airplane(q)" state (waiting to
     * request landing).  This encodes the correct slot assignment: plane p should
     * leave heights 0..c-1 free for the c planes that will land ahead of it.
     */
    private float[] computeAT(ExtendedTransition t) {
        String   action = t.action();
        String[] parts  = t.from().split("\\|");

        boolean isApproach = action.startsWith("approach[");
        boolean isDescend  = action.startsWith("descend[");
        int p = isDescend ? t.extractFirstIndex() : 0;
        int h = isDescend ? t.extractLastIndex()  : 0;

        boolean targetEmpty = false;
        if (isDescend) {
            int hIdx = 2 + h;
            if (hIdx < parts.length) targetEmpty = "Empty".equals(parts[hIdx]);
        }

        boolean consecutive   = heightsAreConsecutive(parts);
        float   heightIdxNorm = isDescend && k > 0 ? h / (float) k : 0f;

        int lowerWaitingCount = 0;
        if (isDescend) {
            for (int q = 0; q < p; q++) {
                int qIdx = 2 + k + q;
                if (qIdx < parts.length && parts[qIdx].equals("Airplane(" + q + ")")) {
                    lowerWaitingCount++;
                }
            }
        }
        boolean heightMatchesSlot = isDescend && (h == lowerWaitingCount);

        return new float[] {
            isApproach        ? 1f : 0f,
            targetEmpty       ? 1f : 0f,
            consecutive       ? 1f : 0f,
            heightIdxNorm,
            heightMatchesSlot ? 1f : 0f
        };
    }

    private boolean heightsAreConsecutive(String[] parts) {
        boolean seenEmpty = false;
        for (int h = 0; h < k; h++) {
            int idx = 2 + h;
            if (idx >= parts.length) break;
            String slot = parts[idx];
            if ("Empty".equals(slot)) {
                seenEmpty = true;
            } else if (slot.startsWith("Occupied[") && seenEmpty) {
                return false;
            }
        }
        return true;
    }

    // ── DP ────────────────────────────────────────────────────────────────────

    /**
     * DP features (5):
     * is_take, phil_is_ready, phil_is_hungry, is_min_ready_idx, is_min_hungry_idx.
     *
     * <p>State layout: 3 components per philosopher at positions 3i, 3i+1, 3i+2.
     * parts[3i] = philosopher i sub-state ("Hungry", "Ready", "Thinking", …).
     */
    private float[] computeDP(ExtendedTransition t) {
        String   action = t.action();
        String[] parts  = t.from().split("\\|");

        boolean isTake = action.startsWith("take[");
        int     i      = isTake ? t.extractFirstIndex() : 0;

        boolean philReady   = false;
        boolean philHungry  = false;
        boolean isMinReady  = false;
        boolean isMinHungry = false;
        if (isTake) {
            int idx = 3 * i;
            if (idx < parts.length) {
                philReady  = "Ready".equals(parts[idx]);
                String ms  = (idx + 2 < parts.length) ? parts[idx + 2] : "";
                philHungry = "Hungry".equals(parts[idx]) && !"Done".equals(ms);
            }
            isMinReady  = true;
            isMinHungry = true;
            for (int j = 0; j < i; j++) {
                int jIdx = 3 * j;
                if (jIdx < parts.length) {
                    if ("Ready".equals(parts[jIdx]))  isMinReady  = false;
                    String jms = (jIdx + 2 < parts.length) ? parts[jIdx + 2] : "";
                    if ("Hungry".equals(parts[jIdx]) && !"Done".equals(jms)) isMinHungry = false;
                }
            }
        }

        return new float[] {
            isTake      ? 1f : 0f,
            philReady   ? 1f : 0f,
            philHungry  ? 1f : 0f,
            isMinReady  ? 1f : 0f,
            isMinHungry ? 1f : 0f
        };
    }

    // ── BW ────────────────────────────────────────────────────────────────────

    /**
     * BW features (7):
     * is_approve, is_refuse, is_assign, doc_is_rejected,
     * crew_is_pending, crew_is_rejected, is_min_eligible_assign.
     *
     * <p>State layout: parts[0]=Document state, parts[t+1]=Crew(t) state.
     */
    private float[] computeBW(ExtendedTransition t) {
        String   action = t.action();
        String[] parts  = t.from().split("\\|");

        boolean isApprove = "approve".equals(action);
        boolean isRefuse  = "refuse".equals(action);
        boolean isAssign  = action.startsWith("assign[");
        int     crewIdx   = isAssign ? t.extractFirstIndex() : 0;

        boolean docRejected   = parts.length > 0 && "Rejected".equals(parts[0]);
        boolean crewPending   = false;
        boolean crewRejected  = false;
        boolean isMinEligible = false;
        if (isAssign) {
            int cIdx = crewIdx + 1;
            if (cIdx < parts.length) {
                crewPending  = "Pending".equals(parts[cIdx]);
                crewRejected = parts[cIdx].startsWith("Rejected");
            }
            isMinEligible = true;
            for (int j = 0; j < crewIdx; j++) {
                int jIdx = j + 1;
                if (jIdx < parts.length) {
                    String cs = parts[jIdx];
                    if ("Pending".equals(cs) || cs.startsWith("Rejected")) {
                        isMinEligible = false;
                        break;
                    }
                }
            }
        }

        return new float[] {
            isApprove     ? 1f : 0f,
            isRefuse      ? 1f : 0f,
            isAssign      ? 1f : 0f,
            docRejected   ? 1f : 0f,
            crewPending   ? 1f : 0f,
            crewRejected  ? 1f : 0f,
            isMinEligible ? 1f : 0f
        };
    }

    // ── CM ────────────────────────────────────────────────────────────────────

    /**
     * CM features (3):
     * dist_to_safe_norm, is_min_dist_candidate, is_min_mouse_among_min_dist.
     *
     * <p>Action: {@code mouse[i][move[j]]} — mouse i moves to position j.
     * safePlace = floor((2k+1)/2).
     */
    private float[] computeCM(ExtendedTransition t) {
        int pos   = t.extractLastIndex();
        int mouse = t.extractFirstIndex();
        int dist  = Math.abs(pos - safePlace);

        boolean isMinDist  = dist == cachedMinDist;
        boolean isMinMouse = isMinDist && mouse == cachedMinDistMouse;

        return new float[] {
            safePlace > 0 ? dist / (float) safePlace : 0f,
            isMinDist  ? 1f : 0f,
            isMinMouse ? 1f : 0f
        };
    }

    // ── TL ────────────────────────────────────────────────────────────────────

    /**
     * TL features (2):
     * dest_is_explored, is_max_index.
     *
     * <p>Action: {@code get[id]}.
     */
    private float[] computeTL(ExtendedTransition t) {
        boolean explored = ctx != null && ctx.succMap.containsKey(t.to());
        boolean isGet    = t.action().startsWith("get[");
        boolean isMax    = isGet && t.extractFirstIndex() == cachedMaxGetIdx;

        return new float[] {
            explored ? 1f : 0f,
            isMax    ? 1f : 0f
        };
    }

    // ── TA ────────────────────────────────────────────────────────────────────

    /**
     * TA features (6):
     * is_agency_succ, is_agency_fail, is_purchase, all_monitors_success,
     * this_monitor_success, purchase_to_disallow_dist_norm.
     *
     * <p>State layout: parts[0]=Agency, parts[1]=AgencyMonitor,
     * parts[2+2i]=Service(i), parts[3+2i]=ServiceMonitor(i).
     */
    private float[] computeTA(ExtendedTransition t) {
        String   action = t.action();
        String[] parts  = t.from().split("\\|");

        boolean isSucc     = "agency.succ".equals(action);
        boolean isFail     = "agency.fail".equals(action);
        boolean isPurchase = action.startsWith("purchase[");
        int     pIdx       = isPurchase ? t.extractFirstIndex() : 0;

        boolean allSuccess = true;
        for (int i = 0; i < n; i++) {
            int idx = 3 + 2 * i;
            if (idx >= parts.length || !"Success".equals(parts[idx])) { allSuccess = false; break; }
        }

        boolean thisMonSuccess = false;
        if (isPurchase) {
            int idx = 3 + 2 * pIdx;
            if (idx < parts.length) thisMonSuccess = "Success".equals(parts[idx]);
        }

        int disallowJ = 0;
        if (parts.length > 1 && parts[1].startsWith("Disallow[")) {
            int b1 = parts[1].indexOf('[');
            int b2 = parts[1].indexOf(']', b1);
            if (b1 >= 0 && b2 > b1) {
                try { disallowJ = Integer.parseInt(parts[1].substring(b1 + 1, b2)); }
                catch (NumberFormatException ignored) {}
            }
        }

        return new float[] {
            isSucc         ? 1f : 0f,
            isFail         ? 1f : 0f,
            isPurchase     ? 1f : 0f,
            allSuccess     ? 1f : 0f,
            thisMonSuccess ? 1f : 0f,
            (isPurchase && n > 0) ? Math.abs(pIdx - disallowJ) / (float) n : 0f
        };
    }

    // ── metadata ──────────────────────────────────────────────────────────────

    @Override
    public List<String> getFeatureNames() {
        return switch (family) {
            case "AT" -> List.of(
                "is_approach", "target_height_empty", "heights_consecutive", "height_idx_norm",
                "height_matches_approach_slot");
            case "DP" -> List.of(
                "is_take", "phil_is_ready", "phil_is_hungry",
                "is_min_ready_idx", "is_min_hungry_idx");
            case "BW" -> List.of(
                "is_approve", "is_refuse", "is_assign", "doc_is_rejected",
                "crew_is_pending", "crew_is_rejected", "is_min_eligible_assign");
            case "CM" -> List.of(
                "dist_to_safe_norm", "is_min_dist_candidate", "is_min_mouse_among_min_dist");
            case "TL" -> List.of("dest_is_explored", "is_max_index");
            case "TA" -> List.of(
                "is_agency_succ", "is_agency_fail", "is_purchase",
                "all_monitors_success", "this_monitor_success",
                "purchase_to_disallow_dist_norm");
            default -> List.of();
        };
    }

    @Override
    public String getFeatureGroupName() {
        return "SUPER_CUSTOM";
    }
}
