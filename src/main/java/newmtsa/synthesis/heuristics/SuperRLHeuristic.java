package newmtsa.synthesis.heuristics;

import newmtsa.synthesis.Director;
import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.OTFDirectedControlledSynthesis;
import newmtsa.synthesis.features.FeatureCompute;

import java.util.*;

/**
 * SuperDFS heuristic adapted for Python-driven RL training.
 *
 * Extends SuperHeuristic so all DFS state and traversal logic are inherited.
 * pickControllable() returns null when a multi-controllable decision is needed,
 * causing pick() to return -1 as a sentinel. driveToDecision() detects this
 * and surfaces feature vectors to the Python agent.
 *
 * Protocol (called from DCSForPython):
 *   1. getFrontierWithFeatures() → driveToDecision() — auto-expands until agent must decide;
 *      returns SUPER feature vectors for the controllable candidates, or empty when done.
 *   2. Python agent picks an index.
 *   3. expand(index) → expandChoice(index) — commits the choice, resumes DFS.
 *   4. Repeat from 1.
 *
 * bind() must be called once after the engine is built.
 */
public class SuperRLHeuristic extends SuperHeuristic {

    private OTFDirectedControlledSynthesis engine;
    private FeatureCompute                 featureCompute;

    private List<ExtendedTransition> rlCandidates       = Collections.emptyList();
    private int                      lastCtrlExpanded    = 0;
    private int                      lastNonCtrlExpanded = 0;

    public void bind(OTFDirectedControlledSynthesis engine, FeatureCompute featureCompute) {
        this.engine         = engine;
        this.featureCompute = featureCompute;
    }

    // ── RL interface ──────────────────────────────────────────────────────────

    /**
     * Auto-expands all non-decision steps via the inherited pick() loop until the RL
     * agent must choose among multiple controllable transitions.
     *
     * @return SUPER feature vectors (one per candidate) at the decision point,
     *         or an empty list when exploration has ended.
     */
    public List<float[]> driveToDecision() {
        // Do NOT reset lastCtrlExpanded/lastNonCtrlExpanded here — they carry
        // the count from the preceding expandChoice() call so the Python side
        // sees agent-choice + auto-expanded in one combined step total.
        rlCandidates = Collections.emptyList();

        while (!engine.isExplorationEnded()) {
            List<ExtendedTransition> frontier = engine.getFrontier();
            if (frontier.isEmpty()) {
                System.err.println("[SuperRLHeuristic] BUG: frontier empty but isExplorationEnded()=false"
                    + " | state=" + currentState
                    + " | stack=" + stack.size()
                    + " | ignored=" + ignored.size());
                break;
            }

            int idx = pick(frontier);

            if (idx == -1) {
                List<float[]> feats = new ArrayList<>(rlCandidates.size());
                for (ExtendedTransition t : rlCandidates) feats.add(featureCompute.compute(t));
                return feats;
            }

            ExtendedTransition t = frontier.get(idx);
            if (t.isControllable(controllable)) lastCtrlExpanded++;
            else lastNonCtrlExpanded++;
            engine.expand(idx);
        }

        rlCandidates = Collections.emptyList();
        return Collections.emptyList();
    }

    /**
     * Commits the candidate chosen by the RL agent at choiceIdx.
     * All other candidates are pushed onto the ignored list.
     */
    public void expandChoice(int choiceIdx) {
        if (rlCandidates.isEmpty()) return;
        lastCtrlExpanded    = 0;
        lastNonCtrlExpanded = 0;

        ExtendedTransition chosen = rlCandidates.get(choiceIdx);
        for (int i = 0; i < rlCandidates.size(); i++) {
            if (i != choiceIdx) ignored.add(rlCandidates.get(i));
        }
        decidedStates.add(currentState);

        List<ExtendedTransition> frontier = engine.getFrontier();
        int idx = findInPending(frontier, chosen);
        if (idx >= 0) {
            engine.expand(idx);
            currentState = chosen.to();
            if (ctx.getFutureAddToFrontier(chosen).isEmpty()) forceStackPop = true;
            lastCtrlExpanded = 1;
        }
        rlCandidates = Collections.emptyList();
    }

    public int getLastCtrlExpanded()    { return lastCtrlExpanded; }
    public int getLastNonCtrlExpanded() { return lastNonCtrlExpanded; }

    // ── pickControllable ──────────────────────────────────────────────────────

    @Override
    protected ExtendedTransition pickControllable(List<ExtendedTransition> ctrl, boolean canReturnNull) {
        if (canReturnNull) return null;
        // Signal RL decision point: store candidates, return null so pick() returns -1.
        this.rlCandidates = ctrl;
        return null;
    }

    @Override
    public void notifyExplorationEnd(Director result) {}
}
