package newmtsa.synthesis;

import newmtsa.parser.FSPParser;
import newmtsa.parser.ast.ControllerSpecDef;
import newmtsa.parser.ast.FSPModel;
import newmtsa.parser.ast.LTS;
import newmtsa.parser.ast.LtlPropertyDef;
import newmtsa.synthesis.features.FeatureCompute;
import newmtsa.synthesis.features.FeatureType;
import newmtsa.synthesis.gr1.OTFDirectedControledSyntesisGR1;
import newmtsa.synthesis.heuristics.Heuristic;
import newmtsa.synthesis.heuristics.HeuristicType;
import newmtsa.synthesis.heuristics.SuperRLHeuristic;
import newmtsa.synthesis.nonblocking.OTFDirectedControledSyntesisNonBlocking;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Step-by-step interface to the DCS algorithm for use from Python (e.g. via JPype).
 * Supports both non-blocking and GR(1) synthesis modes.
 *
 * <p>The Python side drives exploration by repeatedly calling {@link #expand(int)} with
 * the chosen frontier index, making this class suitable as a Reinforcement Learning
 * environment.
 *
 * <p>Use {@link #getSynthesisType()} to determine which mode the instance is running.
 * All exploration calls are dispatched through the common {@link OTFDirectedControlledSynthesis}
 * interface; {@link #gr1Engine} and {@link #nbEngine} expose the typed engines for callers
 * that need mode-specific API.
 */
public class DCSForPython {

    /** Whether this instance is running a non-blocking or a GR(1) synthesis. */
    public enum SynthesisType { NON_BLOCKING, GR1 }

    // ── instance identity ─────────────────────────────────────────────────────

    public final String        instanceName;
    public final SynthesisType synthesisType;

    // ── engines ───────────────────────────────────────────────────────────────

    /** Polymorphic engine — all common exploration calls go here. */
    public final OTFDirectedControlledSynthesis engine;

    /** GR(1) engine; {@code null} when {@link SynthesisType#NON_BLOCKING}. */
    public final OTFDirectedControledSyntesisGR1 gr1Engine;

    /** Non-blocking engine; {@code null} when {@link SynthesisType#GR1}. */
    public final OTFDirectedControledSyntesisNonBlocking nbEngine;

    // ── inputs (mirrored from engine for direct Python access) ────────────────

    public final List<LTS>   components;
    public final Set<String> controllable;
    public final Heuristic   heuristic;

    // ── constructors ──────────────────────────────────────────────────────────

    /**
     * Non-blocking convenience constructor — no feature computation.
     */
    public DCSForPython(String               instanceName,
                        List<LTS>            components,
                        List<LtlPropertyDef> safetyProperties,
                        Set<String>          markingActions,
                        Set<String>          controllable,
                        Heuristic            heuristic) {
        this(instanceName, components, safetyProperties, markingActions, controllable, heuristic, null);
    }

    /**
     * Non-blocking full constructor — optionally enables per-transition feature computation.
     */
    public DCSForPython(String               instanceName,
                        List<LTS>            components,
                        List<LtlPropertyDef> safetyProperties,
                        Set<String>          markingActions,
                        Set<String>          controllable,
                        Heuristic            heuristic,
                        FeatureCompute       featureCompute) {
        this(instanceName, components, safetyProperties, markingActions, controllable,
             heuristic, featureCompute, false);
    }

    public DCSForPython(String               instanceName,
                        List<LTS>            components,
                        List<LtlPropertyDef> safetyProperties,
                        Set<String>          markingActions,
                        Set<String>          controllable,
                        Heuristic            heuristic,
                        FeatureCompute       featureCompute,
                        boolean              frontierRestriction) {
        this.instanceName  = instanceName;
        this.synthesisType = SynthesisType.NON_BLOCKING;
        this.gr1Engine     = null;

        this.nbEngine = new OTFDirectedControledSyntesisNonBlocking(
                components, safetyProperties, markingActions, controllable,
                heuristic, false, Integer.MAX_VALUE, false, featureCompute, frontierRestriction);

        this.engine       = this.nbEngine;
        this.components   = nbEngine.getComponents();
        this.controllable = nbEngine.getControllable();
        this.heuristic    = heuristic;
    }

    /**
     * GR(1) constructor — wraps {@link OTFDirectedControledSyntesisGR1}.
     * No feature computation.
     */
    public DCSForPython(String               instanceName,
                        List<LTS>            components,
                        List<LtlPropertyDef> assumptions,
                        List<LtlPropertyDef> guarantees,
                        Set<String>          controllable,
                        Heuristic            heuristic) {
        this(instanceName, components, assumptions, guarantees, controllable, heuristic, null);
    }

    /**
     * GR(1) constructor with optional feature computation.
     */
    public DCSForPython(String               instanceName,
                        List<LTS>            components,
                        List<LtlPropertyDef> assumptions,
                        List<LtlPropertyDef> guarantees,
                        Set<String>          controllable,
                        Heuristic            heuristic,
                        FeatureCompute       featureCompute) {
        this(instanceName, components, assumptions, guarantees, controllable,
             heuristic, featureCompute, false);
    }

    public DCSForPython(String               instanceName,
                        List<LTS>            components,
                        List<LtlPropertyDef> assumptions,
                        List<LtlPropertyDef> guarantees,
                        Set<String>          controllable,
                        Heuristic            heuristic,
                        FeatureCompute       featureCompute,
                        boolean              frontierRestriction) {
        if (controllable.isEmpty())
            throw new IllegalArgumentException("DCS requires at least one controllable action");

        this.instanceName  = instanceName;
        this.synthesisType = SynthesisType.GR1;
        this.heuristic     = heuristic;
        this.nbEngine      = null;
        this.gr1Engine     = new OTFDirectedControledSyntesisGR1(
                components, assumptions, guarantees, controllable, heuristic, false, false,
                featureCompute, frontierRestriction);

        this.engine       = this.gr1Engine;
        this.components   = gr1Engine.getComponents();
        this.controllable = gr1Engine.getControllable();
    }

    // ── per-step expansion counts (non-SuperRL; SuperRL delegates to heuristic) ─
    private int lastCtrlExpanded    = 0;
    private int lastNonCtrlExpanded = 0;

    // ── public API ────────────────────────────────────────────────────────────

    /** Returns whether this instance is running a non-blocking or a GR(1) synthesis. */
    public SynthesisType getSynthesisType() {
        return synthesisType;
    }

    /** Returns the name of this synthesis instance. */
    public String getInstanceName() {
        return instanceName;
    }

    /** Returns the full action alphabet of the parallel composition. */
    public Set<String> getAlphabet() { return engine.getAlphabet(); }

    /** Returns the component list used in the parallel composition. */
    public List<LTS> getComponents() { return components; }

    /**
     * Returns the current frontier: the list of pending transitions (from a None state)
     * that have not yet been expanded.
     */
    public List<ExtendedTransition> getFrontier() { return engine.getFrontier(); }

    /** Returns {@code true} when no further expansion is possible. */
    public boolean isExplorationEnded() { return engine.isExplorationEnded(); }

    /**
     * Performs one step of the DCS main loop by expanding the transition at position
     * {@code index} in the current frontier.
     * When the heuristic is {@link SuperRLHeuristic}, {@code index} is interpreted
     * as an index into the last candidate list returned by {@link #getFrontierWithFeatures()}.
     */
    public void expand(int index) {
        if (heuristic instanceof SuperRLHeuristic rl) {
            rl.expandChoice(index);
        } else {
            List<ExtendedTransition> frontier = engine.getFrontier();
            if (index < frontier.size()) {
                boolean ctrl    = frontier.get(index).isControllable(controllable);
                lastCtrlExpanded    = ctrl ? 1 : 0;
                lastNonCtrlExpanded = ctrl ? 0 : 1;
            }
            engine.expand(index);
        }
    }

    /** Controllable transitions expanded in the last agent step (including SuperDFS auto-steps). */
    public int getLastCtrlExpanded() {
        if (heuristic instanceof SuperRLHeuristic rl) return rl.getLastCtrlExpanded();
        return lastCtrlExpanded;
    }

    /** Non-controllable transitions expanded in the last agent step (including SuperDFS auto-steps). */
    public int getLastNonCtrlExpanded() {
        if (heuristic instanceof SuperRLHeuristic rl) return rl.getLastNonCtrlExpanded();
        return lastNonCtrlExpanded;
    }

    /** Total transitions in the director (controller strategy); 0 when unrealizable or not yet finished. */
    public int getDirectorTransitions() {
        Director d = engine.getSynthesisResult();
        if (d == null || !d.isRealizable()) return 0;
        int count = 0;
        for (var list : d.enabled().values()) count += list.size();
        return count;
    }

    /**
     * Returns feature vectors for the current frontier.
     * When the heuristic is {@link SuperRLHeuristic}, auto-drives all non-decision
     * steps first and returns SUPER features only for the controllable candidates
     * at the next decision point (or an empty list when the episode has ended).
     */
    public List<float[]> getFrontierWithFeatures() {
        if (heuristic instanceof SuperRLHeuristic rl) {
            return rl.driveToDecision();
        }
        return engine.getFrontierWithFeatures();
    }

    /** Returns ordered feature names matching positions in feature vectors. Empty list when no FeatureCompute is set. */
    public List<String> getFeatureNames()     { return engine.getFeatureNames(); }

    /** Returns the feature group name (e.g. {@code "BASIC"}, {@code "ROL"}), or {@code null} when no FeatureCompute is set. */
    public String getFeatureGroupName()        { return engine.getFeatureGroupName(); }

    /** Returns the synthesis result. */
    public Director getSynthesisResult() { return engine.getSynthesisResult(); }

    // ── factory ───────────────────────────────────────────────────────────────

    /**
     * Parse an FSP file and construct a ready-to-use {@link DCSForPython}.
     * Automatically selects the non-blocking or GR(1) solver based on the
     * controller spec found in the file.
     */
    public static DCSForPython fromPath(String fspPath,
                                        String heuristicType,
                                        String featureType) throws IOException {
        return fromPath(fspPath, heuristicType, featureType, false);
    }

    public static DCSForPython fromPath(String fspPath,
                                        String heuristicType,
                                        String featureType,
                                        boolean frontierRestriction) throws IOException {
        Path      path  = Paths.get(fspPath);
        FSPModel  model = FSPParser.parse(path);
        String    name  = path.getFileName().toString().replace(".fsp", "");
        Heuristic heuristic = HeuristicType.valueOf(heuristicType.toUpperCase()).create();

        Optional<ControllerSpecDef> nbSpec = model.controllerSpecs().stream()
            .filter(ControllerSpecDef::nonblocking)
            .findFirst();

        if (nbSpec.isPresent()) {
            ControllerSpecDef spec = nbSpec.get();
            List<LtlPropertyDef> safetyProps = spec.safety().stream()
                .flatMap(sn -> model.ltlProperties().stream().filter(p -> p.name().equals(sn)))
                .toList();
            FeatureCompute fc = (featureType != null && !featureType.isEmpty())
                ? FeatureType.valueOf(featureType.toUpperCase()).create()
                : null;
            DCSForPython nb = new DCSForPython(name,
                new ArrayList<>(model.processes()),
                safetyProps,
                new HashSet<>(spec.marking()),
                new HashSet<>(spec.controllable()),
                heuristic, fc, frontierRestriction);
            if (heuristic instanceof SuperRLHeuristic rl) rl.bind(nb.engine, fc);
            return nb;
        }

        // GR(1) / blocking path
        ControllerSpecDef spec = model.controllerSpecs().stream()
            .findFirst()
            .orElseThrow(() -> new IllegalArgumentException(
                "No controller spec found in: " + fspPath));

        List<LtlPropertyDef> guarantees = new ArrayList<>();
        for (String n : spec.liveness()) {
            model.asserts().stream().filter(p -> p.name().equals(n)).findFirst()
                .or(() -> model.fluents().stream().filter(fl -> fl.name().equals(n))
                    .map(fl -> new LtlPropertyDef(fl.name(), List.of(), fl)).findFirst())
                .ifPresent(guarantees::add);
        }

        List<LtlPropertyDef> assumptions = new ArrayList<>();
        for (String n : spec.assumption()) {
            model.asserts().stream().filter(p -> p.name().equals(n)).findFirst()
                .or(() -> model.fluents().stream().filter(fl -> fl.name().equals(n))
                    .map(fl -> new LtlPropertyDef(fl.name(), List.of(), fl)).findFirst())
                .ifPresent(assumptions::add);
        }

        List<LTS> components = new ArrayList<>(model.processes());
        for (LtlPropertyDef g : guarantees) components.add(g.lts());
        for (LtlPropertyDef a : assumptions) {
            if (components.stream().noneMatch(c -> c.name().equals(a.name())))
                components.add(a.lts());
        }

        FeatureCompute fc = (featureType != null && !featureType.isEmpty())
            ? FeatureType.valueOf(featureType.toUpperCase()).create()
            : null;
        DCSForPython gr1 = new DCSForPython(name, components, assumptions, guarantees,
            new HashSet<>(spec.controllable()), heuristic, fc, frontierRestriction);
        if (heuristic instanceof SuperRLHeuristic rl) rl.bind(gr1.engine, fc);
        return gr1;
    }

    // ── additional observable state ───────────────────────────────────────────

    /** Returns {@code true} when this instance runs non-blocking (not GR(1)) synthesis. */
    public boolean isNonBlocking() {
        return synthesisType == SynthesisType.NON_BLOCKING;
    }

    /** Returns {@code true} when this instance runs GR(1) (not non-blocking) synthesis. */
    public boolean isGR1() {
        return synthesisType == SynthesisType.GR1;
    }

    /** Returns {@code true} iff the initial state has been classified as realizable. */
    public boolean isRealizable() { return engine.isRealizable(); }

    /** Number of transitions expanded so far. */
    public int getTransitionsExplored() { return engine.getTransitionsExplored(); }

    /** Number of composite states discovered so far. */
    public int getStatesExplored() { return engine.getStatesExplored(); }
}
