package newmtsa.synthesis.heuristics;

import newmtsa.synthesis.Director;
import newmtsa.synthesis.ExtendedTransition;

import java.util.*;

/**
 * Abstract base for the Super-family of DFS heuristics (SuperDFS, SuperHuman, SuperRL).
 *
 * All DFS state and the full pick() loop live here. Subclasses implement only
 * pickControllable() to supply the strategy for choosing among controllable transitions.
 *
 * pick() returns -1 when pickControllable() returns null for a non-Case-5 decision
 * (canReturnNull=false). This sentinel is used by SuperRLHeuristic.driveToDecision()
 * to detect that the RL agent must choose; other subclasses never return null there.
 */
public abstract class SuperHeuristic implements Heuristic {

    protected SynthesisContext ctx;
    protected Set<String>      controllable;

    protected String                          currentState   = null;
    protected final Deque<ExtendedTransition> stack          = new ArrayDeque<>();
    protected final List<ExtendedTransition>  ignored        = new ArrayList<>();
    protected final Set<String>               decidedStates  = new HashSet<>();

    protected int     stepCount         = 0;
    protected boolean forceStackPop     = false;
    protected int     ignoredExpansions = 0;

    @Override
    public void init(SynthesisContext ctx) {
        this.ctx          = ctx;
        this.controllable = ctx.controllable();
    }

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
                ExtendedTransition ctrlChoice = pickControllable(ctrl, true);
                if (ctrlChoice != null) {
                    chosen = ctrlChoice;
                    for (int i = nonCtrl.size() - 1; i >= 0; i--) stack.push(nonCtrl.get(i));
                    List<ExtendedTransition> others = new ArrayList<>();
                    for (ExtendedTransition t : ctrl) if (t != ctrlChoice) others.add(t);
                    onControllableChosen(ctrlChoice, others);
                    ignored.addAll(others);
                    decidedStates.add(currentState);
                } else {
                    chosen = nonCtrl.get(0);
                    for (int i = nonCtrl.size() - 1; i >= 1; i--) stack.push(nonCtrl.get(i));
                }
            } else {
                chosen = nonCtrl.get(0);
                for (int i = nonCtrl.size() - 1; i >= 1; i--) stack.push(nonCtrl.get(i));
            }
        } else if (!ctrl.isEmpty()) {
            if (decidedStates.contains(currentState)) {
                return pickFromStackOrIgnored(pending);
            }
            chosen = pickControllable(ctrl, false);
            if (chosen == null) return -1;
            if (ctrl.size() > 1) {
                ExtendedTransition finalChosen = chosen;
                List<ExtendedTransition> others = new ArrayList<>();
                for (ExtendedTransition t : ctrl) if (t != finalChosen) others.add(t);
                onControllableChosen(chosen, others);
                ignored.addAll(others);
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

    /**
     * Choose among controllable transitions from the current state.
     *
     * canReturnNull=true  (Case 5: mixed nonctrl+ctrl): return null to expand noncontrollable instead.
     * canReturnNull=false (only controllables): return null only as an RL sentinel (SuperRL);
     *                     all other subclasses must return a non-null transition.
     */
    protected abstract ExtendedTransition pickControllable(List<ExtendedTransition> ctrl, boolean canReturnNull);

    /** Called when a controllable transition is chosen and others are deferred. Subclasses may log. */
    protected void onControllableChosen(ExtendedTransition chosen, List<ExtendedTransition> others) {}

    @Override
    public void notifyExplorationEnd(Director result) {}

    // ── shared helpers ────────────────────────────────────────────────────────

    protected int pickFromStackOrIgnored(List<ExtendedTransition> pending) {
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

    protected boolean goesToError(ExtendedTransition t) {
        if (ctx != null && ctx.errors().contains(t.to())) return true;
        for (String part : t.to().split("\\|")) {
            if ("ERROR".equals(part)) return true;
        }
        return false;
    }

    protected int findInPending(List<ExtendedTransition> pending, ExtendedTransition t) {
        for (int i = 0; i < pending.size(); i++) {
            ExtendedTransition p = pending.get(i);
            if (p.from().equals(t.from()) && p.action().equals(t.action()) && p.to().equals(t.to())) {
                return i;
            }
        }
        return -1;
    }
}
