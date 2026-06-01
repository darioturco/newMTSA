package newmtsa.synthesis.heuristics;

import ai.onnxruntime.*;
import newmtsa.parser.ast.LTS;
import newmtsa.synthesis.Director;
import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.OTFDirectedControlledSynthesis;
import newmtsa.synthesis.features.FeatureCompute;
import newmtsa.synthesis.features.FeatureType;
import newmtsa.synthesis.features.FeaturesContext;

import java.util.*;

/**
 * SuperDFS heuristic with two modes:
 *
 * <p><b>Training mode</b> (default constructor): Python-driven. pickControllable() returns null
 * as a sentinel causing pick() to return -1. driveToDecision() surfaces feature vectors to the
 * Python agent; expandChoice() commits the agent's decision. bind() must be called once.
 *
 * <p><b>Deploy mode</b> (onnxPath constructor): standalone ONNX inference. pickControllable()
 * scores candidates with the loaded model and returns the highest-scoring transition. No Python
 * interaction required.
 */
public class SuperRLHeuristic extends SuperHeuristic implements AutoCloseable {

    // ── training-mode state ───────────────────────────────────────────────────

    private OTFDirectedControlledSynthesis engine;
    private FeatureCompute                 featureCompute;

    private List<ExtendedTransition> rlCandidates       = Collections.emptyList();
    private int                      lastCtrlExpanded    = 0;
    private int                      lastNonCtrlExpanded = 0;

    // ── deploy-mode state ─────────────────────────────────────────────────────

    private final String       onnxPath;
    private OrtEnvironment     ortEnv;
    private OrtSession         ortSession;
    private String             outputName;
    private FeatureCompute     deployFeatures;
    private FeaturesContext    featCtx;

    // ── constructors ──────────────────────────────────────────────────────────

    /** Training mode: Python agent drives decisions via driveToDecision() / expandChoice(). */
    public SuperRLHeuristic() {
        this.onnxPath = null;
    }

    /** Deploy mode: ONNX model at {@code onnxPath} drives controllable picks autonomously. */
    public SuperRLHeuristic(String onnxPath) {
        this.onnxPath = onnxPath;
    }

    // ── init ──────────────────────────────────────────────────────────────────

    @Override
    public void init(SynthesisContext ctx) {
        super.init(ctx);
        if (onnxPath == null) return; // training mode: nothing extra to init

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

        featCtx = new FeaturesContext(
                goals, errors, noneView, succMapProxy, new HashMap<>(),
                ctx.controllable(), Collections.unmodifiableSet(alphabet),
                ctx.components(), ctx.componentMarked());
        deployFeatures = FeatureType.valueOf(
            System.getProperty("feature_type", "SUPER").toUpperCase()).create();
        deployFeatures.init(featCtx);

        try {
            ortEnv     = OrtEnvironment.getEnvironment();
            ortSession = ortEnv.createSession(onnxPath, new OrtSession.SessionOptions());
            Set<String> inputNames = ortSession.getInputNames();
            if (!inputNames.contains("features"))
                throw new IllegalArgumentException("ONNX model missing 'features' input: " + onnxPath);
            outputName = ortSession.getOutputNames().iterator().next();
        } catch (OrtException e) {
            throw new RuntimeException("Failed to load ONNX model: " + onnxPath, e);
        }
    }

    // ── pickControllable ──────────────────────────────────────────────────────

    @Override
    protected ExtendedTransition pickControllable(List<ExtendedTransition> ctrl, boolean canReturnNull) {
        if (onnxPath != null) {
            // deploy mode: ONNX picks
            // canReturnNull=true means mixed state (nonCtrl present) — defer to nonCtrl, exactly as training mode does
            if (canReturnNull) return null;
            if (ctrl.size() == 1 || ortSession == null) return ctrl.get(0);
            try {
                deployFeatures.precompute(ctrl);
                int N = ctrl.size();
                int F = deployFeatures.compute(ctrl.get(0)).length;
                float[][] data = new float[N][F];
                for (int i = 0; i < N; i++) data[i] = deployFeatures.compute(ctrl.get(i));
                try (OnnxTensor tensor = OnnxTensor.createTensor(ortEnv, data);
                     OrtSession.Result res = ortSession.run(Map.of("features", tensor))) {
                    float[] scores = (float[]) res.get(outputName).orElseThrow().getValue();
                    int best = 0;
                    for (int i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
                    if (Boolean.getBoolean("rl.verbose")) {
                        System.out.println("[RL] " + N + " candidate(s):");
                        for (int i = 0; i < N; i++) {
                            System.out.printf("  [%d] %s | score=%7.4f | feat=%s%s%n",
                                i, ctrl.get(i).format(), scores[i],
                                java.util.Arrays.toString(data[i]),
                                i == best ? "  <- CHOSEN" : "");
                        }
                    }
                    ExtendedTransition chosen = ctrl.get(best);
                    featCtx.lastExpandedFrom = chosen.from();
                    featCtx.lastExpandedTo   = chosen.to();
                    return chosen;
                }
            } catch (OrtException e) {
                throw new RuntimeException("ONNX inference failed: " + e.getMessage(), e);
            }
        }

        // training mode: signal RL decision point
        if (canReturnNull) return null;
        this.rlCandidates = ctrl;
        return null;
    }

    // ── training-mode interface ───────────────────────────────────────────────

    public void bind(OTFDirectedControlledSynthesis engine, FeatureCompute featureCompute) {
        this.engine         = engine;
        this.featureCompute = featureCompute;
    }

    /**
     * Auto-expands all non-decision steps until the RL agent must choose among
     * multiple controllable transitions.
     *
     * @return SUPER feature vectors (one per candidate), or empty when done.
     */
    public List<float[]> driveToDecision() {
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
                featureCompute.precompute(rlCandidates);
                List<float[]> feats = new ArrayList<>(rlCandidates.size());
                for (ExtendedTransition t : rlCandidates) feats.add(featureCompute.compute(t));
                return feats;
            }

            ExtendedTransition t = frontier.get(idx);
            if (t.isControllable(controllable)) lastCtrlExpanded++;
            else lastNonCtrlExpanded++;
            totalExpanded++;
            if (Boolean.getBoolean("rl.verbose")) {
                System.out.printf("[SUPER RL] (%d) Expanded %s%n", totalExpanded, t.format());
            }
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
        totalExpanded++;
        if (Boolean.getBoolean("rl.verbose")) {
            if (rlCandidates.size() > 1) {
                System.out.printf("[SUPER RL] [DECISION] (%d)%n", totalExpanded);
                for (int i = 0; i < rlCandidates.size(); i++) {
                    System.out.printf("  %s%s%n", rlCandidates.get(i).format(),
                        i == choiceIdx ? "  <-- CHOSEN" : "");
                }
            } else {
                System.out.printf("[SUPER RL] (%d) Expanded %s%n", totalExpanded, chosen.format());
            }
        }
        for (int i = 0; i < rlCandidates.size(); i++) {
            if (i != choiceIdx) ignored.add(rlCandidates.get(i));
        }
        if (rlCandidates.size() > 1) decidedStates.add(currentState);

        List<ExtendedTransition> frontier = engine.getFrontier();
        int idx = findInPending(frontier, chosen);
        if (idx >= 0) {
            boolean deadEnd = ctx.getFutureAddToFrontier(chosen).isEmpty();
            engine.expand(idx);
            currentState = chosen.to();
            if (deadEnd) forceStackPop = true;
            lastCtrlExpanded = 1;
        }
        rlCandidates = Collections.emptyList();
    }

    public int getLastCtrlExpanded()    { return lastCtrlExpanded; }
    public int getLastNonCtrlExpanded() { return lastNonCtrlExpanded; }

    // ── lifecycle ─────────────────────────────────────────────────────────────

    @Override
    public void notifyExplorationEnd(Director result) {}

    @Override
    public void close() {
        try { if (ortSession != null) ortSession.close(); } catch (Exception ignored) {}
        try { if (ortEnv    != null) ortEnv.close();     } catch (Exception ignored) {}
    }
}
