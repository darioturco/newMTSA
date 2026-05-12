package newmtsa.synthesis.heuristics.mcts;

import ai.onnxruntime.*;
import newmtsa.parser.ast.LTS;
import newmtsa.parser.ast.Transition;
import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.features.BasicFeatures;
import newmtsa.synthesis.features.FeatureCompute;
import newmtsa.synthesis.features.FeaturesContext;
import newmtsa.synthesis.features.FeatureType;
import newmtsa.synthesis.heuristics.Heuristic;
import newmtsa.synthesis.heuristics.SynthesisContext;

import java.util.*;

/**
 * MCTS heuristic for OTF-DCS guided by a flat (MLP) ONNX model.
 *
 * <p>At each call to {@link #pick}, runs {@code numSimulations} virtual rollouts
 * from the current frontier without touching the real synthesis state.  Each
 * rollout uses the network for:
 * <ul>
 *   <li><b>Policy prior</b>: softmax of raw scores → P(a) for PUCT action selection.</li>
 *   <li><b>Leaf value</b>: min of raw scores over the virtual frontier (pessimistic estimate).</li>
 * </ul>
 *
 * <p>After simulations, the most-visited root action is returned.  The MCTS subtree
 * rooted at that action is reused on the next call (tree reuse), amortising
 * simulation cost across real steps.
 *
 * <p>Only flat (MLP) ONNX models are supported.  An {@link IllegalArgumentException}
 * is thrown at init time if a non-flat model is loaded.
 */
public class MCTSHeuristic implements Heuristic, AutoCloseable {

    // ── MCTS parameters ───────────────────────────────────────────────────────

    private final int    numSimulations;
    private final double cPuct;
    private final int    maxDepth;

    // ── ONNX ─────────────────────────────────────────────────────────────────

    private final String         onnxPath;
    private final FeatureType    featureType;
    private       OrtEnvironment ortEnv;
    private       OrtSession     ortSession;
    private       String         outputName;

    // ── derived from SynthesisContext ─────────────────────────────────────────

    private SynthesisContext                            ctx;
    private List<Map<String, Map<String, String>>>      compTrans;
    private List<Set<String>>                           compAlpha;
    private FeaturesContext                             featCtx;
    private FeatureCompute                              features;

    // ── MCTS tree state (persists across pick() calls) ────────────────────────

    private MCTSNode currentRoot      = null;
    private int      lastChosenAction = -1;

    // ── virtual successor cache (cleared each pick() call) ────────────────────

    private final Map<String, List<ExtendedTransition>> succCache = new HashMap<>();

    // ── constructors ──────────────────────────────────────────────────────────

    public MCTSHeuristic(String onnxPath, String featureType) {
        this(onnxPath, featureType, 50, 1.5, 10);
    }

    public MCTSHeuristic(String onnxPath, String featureType, int numSimulations, double cPuct, int maxDepth) {
        this.onnxPath       = onnxPath;
        this.featureType    = FeatureType.valueOf(featureType);
        this.numSimulations = numSimulations;
        this.cPuct          = cPuct;
        this.maxDepth       = maxDepth;
    }

    // ── Heuristic interface ───────────────────────────────────────────────────

    @Override
    public void init(SynthesisContext ctx) {
        this.ctx = ctx;

        // Reconstruct component transition tables from ctx.components()
        compTrans = new ArrayList<>();
        compAlpha = new ArrayList<>();
        Set<String> alphabet = new LinkedHashSet<>();
        for (LTS lts : ctx.components()) {
            Map<String, Map<String, String>> actMap = new HashMap<>();
            Set<String> acts = new LinkedHashSet<>(lts.actions());
            for (Transition t : lts.transitions()) {
                actMap.computeIfAbsent(t.action(), k -> new HashMap<>()).put(t.from(), t.to());
                acts.add(t.action());
            }
            compTrans.add(actMap);
            compAlpha.add(acts);
            alphabet.addAll(acts);
        }

        // Build a FeaturesContext from live SynthesisContext views.
        // succMap is proxied so BasicFeatures can call containsKey / get.
        // parents is empty (state-label bits will be approximate for real frontier).
        Set<String> goals       = ctx.goals();
        Set<String> errors      = ctx.errors();
        Set<String> explored    = ctx.exploredStates();
        // none = explored - goals - errors, computed on demand via a live-view Set
        Set<String> noneView = new AbstractSet<>() {
            @Override public boolean contains(Object o) {
                return explored.contains(o) && !goals.contains(o) && !errors.contains(o);
            }
            @Override public Iterator<String> iterator() { return Collections.emptyIterator(); }
            @Override public int size() { return 0; }
        };

        Map<String, List<ExtendedTransition>> succMapProxy = new AbstractMap<>() {
            @Override public boolean containsKey(Object key) {
                return explored.contains(key);
            }
            @Override public List<ExtendedTransition> get(Object key) {
                return ctx.successorsOf((String) key);
            }
            @Override public Set<Map.Entry<String, List<ExtendedTransition>>> entrySet() {
                return Collections.emptySet();
            }
        };

        featCtx = new FeaturesContext(
                goals, errors, noneView, succMapProxy, new HashMap<>(),
                ctx.controllable(), Collections.unmodifiableSet(alphabet),
                ctx.components(), ctx.componentMarked());

        features = featureType.create();
        features.init(featCtx);

        // Load ONNX model
        try {
            ortEnv = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
            ortSession = ortEnv.createSession(onnxPath, opts);

            Set<String> inputNames = ortSession.getInputNames();
            if (inputNames.contains("prev_feat") || inputNames.contains("sequence")) {
                throw new IllegalArgumentException(
                    "MCTSHeuristic supports flat (MLP) ONNX models only. " +
                    "Detected non-flat model at: " + onnxPath);
            }
            if (!inputNames.contains("features")) {
                throw new IllegalArgumentException(
                    "ONNX model missing expected input 'features'. Path: " + onnxPath);
            }

            outputName = ortSession.getOutputNames().iterator().next();
        } catch (OrtException e) {
            throw new RuntimeException("Failed to load ONNX model: " + onnxPath, e);
        }
    }

    @Override
    public int pick(List<ExtendedTransition> pending) {
        if (pending.size() == 1) return 0;

        // Tree reuse: advance root to last chosen child if available
        if (currentRoot != null && lastChosenAction >= 0
                && currentRoot.children.containsKey(lastChosenAction)) {
            currentRoot = currentRoot.children.get(lastChosenAction);
        } else {
            float[] scores = queryNetwork(pending);
            currentRoot = MCTSNode.fromScores(pending, scores);
        }

        // Clear virtual successor cache for this real step
        succCache.clear();

        // Run MCTS simulations
        VirtualState vRoot = new VirtualState(
                new HashSet<>(ctx.exploredStates()), pending, null, null);
        for (int i = 0; i < numSimulations; i++) {
            simulate(currentRoot, vRoot, 0);
        }

        int action = currentRoot.mostVisited();
        lastChosenAction = action;

        // Update lastExpanded bits in featCtx for next feature computation
        ExtendedTransition chosen = pending.get(action);
        featCtx.lastExpandedFrom = chosen.from();
        featCtx.lastExpandedTo   = chosen.to();

        return action;
    }

    // ── MCTS simulation ───────────────────────────────────────────────────────

    private double simulate(MCTSNode node, VirtualState vState, int depth) {
        if (depth >= maxDepth || vState.frontier.isEmpty()) {
            return evaluateLeaf(vState);
        }

        int a = node.selectUCB(cPuct);
        // Guard: frontier may be shorter than root's frontier if some transitions vanished
        if (a >= vState.frontier.size()) a = 0;

        ExtendedTransition t    = vState.frontier.get(a);
        VirtualState       next = vState.expand(t, ctx, compTrans, compAlpha, succCache);

        double value;
        if (!node.children.containsKey(a)) {
            if (next.frontier.isEmpty()) {
                value = 1.0;
            } else {
                float[]  scores = queryNetwork(next.frontier, next);
                MCTSNode child  = MCTSNode.fromScores(next.frontier, scores);
                node.children.put(a, child);
                value = leafValue(scores);
            }
        } else {
            value = simulate(node.children.get(a), next, depth + 1);
        }

        node.update(a, value);
        return value;
    }

    private double evaluateLeaf(VirtualState vState) {
        if (vState.frontier.isEmpty()) return 1.0;
        return leafValue(queryNetwork(vState.frontier, vState));
    }

    /** min of raw scores — pessimistic estimate of frontier quality. */
    private static double leafValue(float[] scores) {
        float min = scores[0];
        for (float s : scores) if (s < min) min = s;
        return min;
    }

    // ── ONNX inference ────────────────────────────────────────────────────────

    /** Query network for the real pending frontier (uses live featCtx). */
    private float[] queryNetwork(List<ExtendedTransition> frontier) {
        return queryNetwork(frontier, null);
    }

    /**
     * Query network for {@code frontier}.
     * If {@code vState != null}, temporarily adjust featCtx for virtual context bits
     * (lastExpandedFrom/To) before computing features, then restore.
     */
    private float[] queryNetwork(List<ExtendedTransition> frontier, VirtualState vState) {
        String savedFrom = featCtx.lastExpandedFrom;
        String savedTo   = featCtx.lastExpandedTo;
        if (vState != null) {
            featCtx.lastExpandedFrom = vState.lastFrom;
            featCtx.lastExpandedTo   = vState.lastTo;
        }

        try {
            int N = frontier.size();
            int F = features.compute(frontier.get(0)).length;
            float[][] data = new float[N][F];
            for (int i = 0; i < N; i++) {
                int[] fv = features.compute(frontier.get(i));
                for (int j = 0; j < F; j++) data[i][j] = fv[j];
            }

            try (OnnxTensor tensor = OnnxTensor.createTensor(ortEnv, data);
                 OrtSession.Result res = ortSession.run(Map.of("features", tensor))) {
                return (float[]) res.get(outputName).orElseThrow().getValue();
            }
        } catch (OrtException e) {
            throw new RuntimeException("ONNX inference failed", e);
        } finally {
            featCtx.lastExpandedFrom = savedFrom;
            featCtx.lastExpandedTo   = savedTo;
        }
    }

    // ── AutoCloseable ─────────────────────────────────────────────────────────

    @Override
    public void close() {
        try {
            if (ortSession != null) ortSession.close();
        } catch (OrtException ignored) {}
    }
}
