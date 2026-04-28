package newmtsa.synthesis.features;

import newmtsa.parser.ast.LTS;
import newmtsa.synthesis.ExtendedTransition;

import java.util.*;

/**
 * Live, read-only view of the DCS exploration state passed to {@link FeatureCompute}
 * implementations.
 *
 * <p>All collection fields are live references into {@code DCSForPython}'s private
 * data structures — they reflect the current exploration state without copying.
 * The mutable scalar fields ({@link #lastExpandedFrom}, {@link #lastExpandedTo},
 * {@link #markedStateFound}, {@link #closedWinningLoopsCount}) are written by
 * {@code DCSForPython} and read by feature implementations.
 */
public final class FeaturesContext {

    // ── live exploration state ────────────────────────────────────────────────

    public final Set<String>                            goals;
    public final Set<String>                            errors;
    public final Set<String>                            none;
    public final Map<String, List<ExtendedTransition>>  succMap;
    public final Map<String, Set<String>>               parents;
    public final Set<String>                            controllable;
    public final Set<String>                            alphabet;

    // ── component structure (needed by isMarked / splitState) ─────────────────

    private final List<LTS>         components;
    private final List<Set<String>> compMarked;

    // ── mutable progress state (written by DCSForPython) ─────────────────────

    public String  lastExpandedFrom        = null;
    public String  lastExpandedTo          = null;
    public boolean markedStateFound        = false;
    public int     closedWinningLoopsCount = 0;

    // ── derived, lazily built ─────────────────────────────────────────────────

    private List<String>         noIdxAlphabet;
    private Map<String, Integer> noIdxIndex;

    // ── constructor ───────────────────────────────────────────────────────────

    public FeaturesContext(Set<String>                            goals,
                           Set<String>                            errors,
                           Set<String>                            none,
                           Map<String, List<ExtendedTransition>>  succMap,
                           Map<String, Set<String>>               parents,
                           Set<String>                            controllable,
                           Set<String>                            alphabet,
                           List<LTS>                              components,
                           List<Set<String>>                      compMarked) {
        this.goals        = goals;
        this.errors       = errors;
        this.none         = none;
        this.succMap      = succMap;
        this.parents      = parents;
        this.controllable = controllable;
        this.alphabet     = alphabet;
        this.components   = components;
        this.compMarked   = compMarked;
    }

    // ── public helpers ────────────────────────────────────────────────────────

    /**
     * Returns the alphabet with all digit characters removed, sorted and deduplicated.
     * Result is cached after the first call.
     */
    public List<String> getNoIdxAlphabet() {
        if (noIdxAlphabet == null) buildNoIdxStructures();
        return noIdxAlphabet;
    }

    /** Returns a map from no-index action label → index in {@link #getNoIdxAlphabet()}. */
    public Map<String, Integer> getNoIdxIndex() {
        if (noIdxIndex == null) buildNoIdxStructures();
        return noIdxIndex;
    }

    /** {@code true} iff {@code state} is marked in every component that declares marked states. */
    public boolean isMarked(String state) {
        String[] parts = state.split("\\|", components.size());
        for (int i = 0; i < components.size(); i++) {
            Set<String> marked = compMarked.get(i);
            if (!marked.isEmpty() && !marked.contains(parts[i])) return false;
        }
        return true;
    }

    /** {@code true} iff {@code state} has been classified (goal, error, or none). */
    public boolean isVisited(String state) {
        return goals.contains(state) || errors.contains(state) || none.contains(state);
    }

    /** Strips all digit characters from {@code label}. */
    public static String removeIndices(String label) {
        return label.replaceAll("[0-9]", "");
    }

    // ── private ───────────────────────────────────────────────────────────────

    private void buildNoIdxStructures() {
        LinkedHashSet<String> set = new LinkedHashSet<>();
        for (String a : alphabet) set.add(removeIndices(a));
        noIdxAlphabet = new ArrayList<>(set);
        Collections.sort(noIdxAlphabet);
        noIdxIndex = new HashMap<>();
        for (int i = 0; i < noIdxAlphabet.size(); i++) {
            noIdxIndex.put(noIdxAlphabet.get(i), i);
        }
    }
}
