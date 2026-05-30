package newmtsa.synthesis.heuristics;

import ai.onnxruntime.*;
import newmtsa.parser.ast.LTS;
import newmtsa.synthesis.ExtendedTransition;
import newmtsa.synthesis.features.BasicFeatures;
import newmtsa.synthesis.features.FeatureCompute;
import newmtsa.synthesis.features.FeaturesContext;
import newmtsa.synthesis.features.FeatureType;

import java.util.*;

/**
 * Reinforcement Learning heuristic for OTF-DCS, guided by a flat (MLP) ONNX
 * policy/value network.
 *
 * <p>Pick rule: compute the feature vector of every frontier transition, run the
 * ONNX model once over the batch, return {@code argmax} of the scalar output.
 *
 * <p>This is the standalone inference counterpart of the Python training loop
 * (DQN/PPO/SAC, flat mode). The same {@link BasicFeatures} pipeline used during
 * training is reused at inference time.
 *
 * <p>When driven from Python via {@code DCSForPython.expand(idx)} this heuristic
 * is bypassed entirely — the Python agent selects the index directly.
 * In that mode the {@code onnxPath} parameter is irrelevant: pass any value
 * (e.g. {@code null}) and the model is never loaded.
 *
 * <p>Only flat (MLP) ONNX models are supported. LSTM / Transformer exports are
 * rejected at {@link #init} time.
 */
public class RLHeuristic implements Heuristic, AutoCloseable {

    private final String         onnxPath;
    private final FeatureType    featureType;
    private       OrtEnvironment ortEnv;
    private       OrtSession     ortSession;
    private       String         outputName;

    private FeaturesContext featCtx;
    private FeatureCompute  features;

    public RLHeuristic(String onnxPath, String featureType) {
        this.onnxPath = onnxPath;
        this.featureType = FeatureType.valueOf(featureType);
    }

    @Override
    public void init(SynthesisContext ctx) {
        if (onnxPath == null || onnxPath.isEmpty()) return;

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
        features = featureType.create();
        features.init(featCtx);

        try {
            ortEnv = OrtEnvironment.getEnvironment();
            ortSession = ortEnv.createSession(onnxPath, new OrtSession.SessionOptions());

            Set<String> inputNames = ortSession.getInputNames();
            if (inputNames.contains("prev_feat") || inputNames.contains("sequence")) {
                throw new IllegalArgumentException(
                    "RLHeuristic supports flat (MLP) ONNX models only. " +
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
        if (ortSession == null)  return 0;

        float[] scores = queryNetwork(pending);
        int best = 0;
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > scores[best]) best = i;
        }

        ExtendedTransition chosen = pending.get(best);
        featCtx.lastExpandedFrom = chosen.from();
        featCtx.lastExpandedTo   = chosen.to();
        return best;
    }

    private float[] queryNetwork(List<ExtendedTransition> frontier) {
        try {
            int N = frontier.size();
            int F = features.compute(frontier.get(0)).length;
            float[][] data = new float[N][F];
            for (int i = 0; i < N; i++) data[i] = features.compute(frontier.get(i));
            try (OnnxTensor tensor = OnnxTensor.createTensor(ortEnv, data);
                 OrtSession.Result res = ortSession.run(Map.of("features", tensor))) {
                return (float[]) res.get(outputName).orElseThrow().getValue();
            }
        } catch (OrtException e) {
            throw new RuntimeException("ONNX inference failed", e);
        }
    }

    @Override
    public void close() {
        try { if (ortSession != null) ortSession.close(); } catch (OrtException ignored) {}
    }
}
