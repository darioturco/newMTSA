package newmtsa.synthesis.features;

import newmtsa.parser.ast.LTS;
import newmtsa.synthesis.ExtendedTransition;

import java.util.*;

/**
 * Extended feature set: all {@link BasicFeatures} plus role-based and action-encoding features
 * adapted from the original {@code DCSFeatureCalculator}.
 *
 * <p>A <em>role</em> is the abstract submachine a component state belongs to — the state name
 * with any {@code [N]} index suffix removed (e.g. {@code Holding[1]} → {@code Holding}).
 *
 * <p>Feature vector layout (total = {@code 3·|Σ̃| + 3·|R̃| + 20}):
 * <ol>
 *   <li><b>BasicFeatures</b> (indices 0..2n+18) — {@code 2·|Σ̃|+19} bits — see {@link BasicFeatures}.</li>
 *   <li><b>abstract_role_parent</b> ({@code |R̃|} values) — ternary count (-1/0/1) of components
 *       where the <em>source</em> composite state belongs to each abstract role
 *       (0 components → -1, 1 component → 0, ≥2 components → 1).</li>
 *   <li><b>abstract_role_child</b> ({@code |R̃|} values) — same for the <em>target</em> state.</li>
 *   <li><b>abstract_role_local</b> ({@code |R̃|} values) — same, restricted to components where the
 *       role <em>changed</em> between source and target (i.e. the transition caused a submachine change).</li>
 *   <li><b>abstract_action_onehot</b> ({@code |Σ̃|} values) — one-hot of the action's abstract label
 *       (index suffix removed) over the sorted no-index alphabet.</li>
 *   <li><b>has_index</b> (1 bit) — 1 if the raw action label contained a {@code [N]} index suffix
 *       that was removed to obtain the abstract label, 0 if the action has no such suffix.</li>
 * </ol>
 */
public class RolFeatures implements FeatureCompute {

    private FeaturesContext ctx;
    private BasicFeatures   basic;

    /** Sorted vocabulary of all abstract roles across all component LTS states. */
    private List<String> abstractRoles;

    /** Number of component LTS processes in the parallel composition (used to split composite states). */
    private int numComponents;

    @Override
    public void init(FeaturesContext context) {
        this.ctx   = context;
        this.basic = new BasicFeatures();
        this.basic.init(context);

        List<LTS> components = context.getComponents();
        this.numComponents   = components.size();

        // Collect abstract roles from every component LTS.
        // abstractRole() strips both [N] index suffixes AND (N) instance-name suffixes
        // so that Monitor(0)/Monitor(1)/... all collapse to the same "Monitor" role.
        Set<String> allRoles = new TreeSet<>();
        for (LTS lts : components) {
            for (String state : lts.states()) {
                allRoles.add(abstractRole(state));
            }
        }
        this.abstractRoles = new ArrayList<>(allRoles);

        ctx.getNoIdxAlphabet(); // pre-compute alphabet structures
    }

    @Override
    public int[] compute(ExtendedTransition t) {
        int[] basicF = basic.compute(t);
        int   n      = ctx.getNoIdxAlphabet().size();
        int   r      = abstractRoles.size();

        // Append role + action features after the BasicFeatures vector.
        int[] f = Arrays.copyOf(basicF, basicF.length + 3 * r + n + 1);
        int   p = basicF.length;

        // Split composite states into per-component parts.
        // GR(1) states carry a "#phase" suffix on the last component — strip it so role
        // lookup produces the correct abstract role name.
        String[] fromParts = splitAndClean(t.from());
        String[] toParts   = splitAndClean(t.to());

        List<String> parentRoles  = new ArrayList<>(numComponents);
        List<String> childRoles   = new ArrayList<>(numComponents);
        // local = only components where the role changed between source and target
        List<String> changedRoles = new ArrayList<>();

        for (int i = 0; i < numComponents; i++) {
            String pRole = abstractRole(fromParts[i]);
            String cRole = abstractRole(toParts[i]);
            parentRoles.add(pRole);
            childRoles.add(cRole);
            if (!pRole.equals(cRole)) changedRoles.add(cRole);
        }

        // abstract_role_parent: ternary count (0/1/2+) per abstract role in source state
        p = fillTernaryRoles(parentRoles,  f, p);
        // abstract_role_child: ternary count per abstract role in target state
        p = fillTernaryRoles(childRoles,   f, p);
        // abstract_role_local: ternary count per abstract role in components that changed role
        p = fillTernaryRoles(changedRoles, f, p);

        // abstract_action_onehot: one-hot over sorted no-index alphabet
        String  abstractAction = FeaturesContext.removeIndices(t.action());
        Integer ai             = ctx.getNoIdxIndex().get(abstractAction);
        if (ai != null) f[p + ai] = 1;
        p += n;

        // has_index: 1 if the action label had a [N] index suffix that was removed, 0 otherwise
        f[p] = abstractAction.equals(t.action()) ? 0 : 1;

        return f;
    }

    @Override
    public List<String> getFeatureNames() {
        List<String> names = new ArrayList<>(basic.getFeatureNames());
        for (String role : abstractRoles) names.add("role_parent_" + role);
        for (String role : abstractRoles) names.add("role_child_"  + role);
        for (String role : abstractRoles) names.add("role_local_"  + role);
        for (String a   : ctx.getNoIdxAlphabet()) names.add("abstract_action_" + a);
        names.add("has_index");
        return names;
    }

    @Override
    public String getFeatureGroupName() {
        return "ROL";
    }

    /**
     * Splits a composite state string by {@code |} into per-component parts and strips
     * any trailing {@code #...} phase suffix (present in GR(1) states).
     */
    private String[] splitAndClean(String composite) {
        String[] parts = composite.split("\\|", numComponents);
        for (int i = 0; i < parts.length; i++) {
            int hash = parts[i].indexOf('#');
            if (hash >= 0) parts[i] = parts[i].substring(0, hash);
        }
        return parts;
    }

    /**
     * Returns the abstract role of a state name by stripping both:
     * - {@code [N]} index suffixes (e.g. {@code Holding[1]} → {@code Holding})
     * - {@code (N)} or {@code (N, M)} instance-name suffixes (e.g. {@code Monitor(0)} → {@code Monitor})
     *
     * The second stripping is needed because processes defined as {@code P(Id=0) = (expr)}
     * get their initial state named after the instance (e.g. {@code "Monitor(0)"}), and
     * these parenthesized suffixes are NOT removed by {@link FeaturesContext#removeIndices}.
     */
    private static String abstractRole(String state) {
        String s = FeaturesContext.removeIndices(state);
        return s.replaceAll("\\(\\d+(?:,\\s*\\d+)*\\)$", "");
    }

    /**
     * Writes a ternary-binned count for each abstract role into {@code f} starting at
     * {@code offset}: 0 = absent in any component, 1 = one component has that role, 2 = two or more.
     *
     * @return next free index in {@code f}
     */
    private int fillTernaryRoles(List<String> roles, int[] f, int offset) {
        // count occurrences of each abstract role, cap at 2, then shift to [-1, 0, 1]
        for (String role : roles) {
            int idx = Collections.binarySearch(abstractRoles, role);
            if (idx >= 0) f[offset + idx] = Math.min(f[offset + idx] + 1, 2);
        }
        for (int i = 0; i < abstractRoles.size(); i++) f[offset + i] -= 1;
        return offset + abstractRoles.size();
    }
}
