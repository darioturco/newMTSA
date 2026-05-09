package newmtsa.synthesis.features;

import newmtsa.synthesis.ExtendedTransition;

import java.util.List;

/**
 * Strategy interface for computing per-transition binary feature vectors during
 * DCS exploration.
 *
 * <p>Register an implementation with
 * {@link newmtsa.synthesis.DCSForPython#DCSForPython(String,
 * java.util.List, java.util.List, java.util.Set, java.util.Set,
 * newmtsa.synthesis.heuristics.Heuristic, FeatureCompute)}.
 * Once registered, every call to
 * {@link newmtsa.synthesis.DCSForPython#getFrontierWithFeatures()}
 * invokes {@link #compute(ExtendedTransition)} for each frontier transition.
 *
 * <p>Implementations may be written in Java or, via JPype, in Python.
 */
public interface FeatureCompute {

    /**
     * Called exactly once after the {@link FeaturesContext} has been built but
     * before any transition is expanded. Use this to pre-compute alphabet-derived
     * data that would otherwise be recomputed on every {@link #compute} call.
     */
    void init(FeaturesContext context);

    /**
     * Returns a binary (0/1) feature vector for the given frontier transition.
     * The returned array must have the same length for every call within one episode.
     */
    int[] compute(ExtendedTransition transition);

    /**
     * Returns an ordered list of names, one per position in the vector returned by
     * {@link #compute}. Must be called after {@link #init}.
     */
    List<String> getFeatureNames();

    /**
     * Returns a short identifier for this feature set (e.g. {@code "BASIC"}, {@code "ROL"}).
     */
    String getFeatureGroupName();
}
