package newmtsa.synthesis;

import newmtsa.parser.ast.LTS;

import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * Common step-by-step DCS engine interface implemented by both the non-blocking
 * ({@link newmtsa.synthesis.nonblocking.OTFDirectedControledSyntesisNonBlocking}) and
 * GR(1) ({@link newmtsa.synthesis.gr1.OTFDirectedControledSyntesisGR1}) engines.
 *
 * <p>{@link DCSForPython} holds a single reference of this type and delegates all
 * exploration calls to it, using {@link DCSForPython.SynthesisType} only where the
 * two modes genuinely differ.
 */
public interface OTFDirectedControlledSynthesis {

    List<LTS>   getComponents();
    Set<String> getControllable();
    Set<String> getAlphabet();

    List<ExtendedTransition> getFrontier();
    boolean                  isExplorationEnded();
    void                     expand(int index);
    List<int[]>              getFrontierWithFeatures();
    SynthesisResult          getSynthesisResult();
    boolean                  isRealizable();
    int                      getTransitionsExplored();
    int                      getStatesExplored();

    /** Returns ordered feature names matching positions in {@link #getFrontierWithFeatures()} vectors. Empty list when no feature compute is set. */
    default List<String> getFeatureNames() { return Collections.emptyList(); }

    /** Returns the feature group name (e.g. {@code "BASIC"}, {@code "ROL"}), or {@code null} when no feature compute is set. */
    default String getFeatureGroupName() { return null; }
}
