package newmtsa.synthesis.features;

/**
 * Catalogue of available feature sets for RL-guided DCS exploration.
 *
 * <p>Use {@link #create()} to obtain a fresh {@link FeatureCompute} instance.
 * From Python pass the enum name as a string and call {@code FeatureType.valueOf(name).create()}.
 *
 * <pre>
 * BASIC – features from "On the fly controller synthesis via reinforcement learning"
 * ROL   – BASIC features + role-based encoding (abstract submachine per component) + one-hot action + has_index
 * SUPER – per-transition scoring features: abstract action, normalised indices, destination state,
 *         role fractions, changed-role encoding, phase — size-normalised for cross-instance generalisation
 * SUPER_GENERAL – SUPER + cross-candidate index ranking (min/max/rank/closest-to-centre per abstract
 *         action). Family-agnostic alternative to SUPER_CUSTOM with no hardcoded substate strings.
 * </pre>
 */
public enum FeatureType {

    BASIC,
    /** BasicFeatures extended with role-based component state encoding — see {@link RolFeatures}. */
    ROL,
    /** Per-transition scoring features designed for cross-instance generalisation — see {@link SuperFeatures}. */
    SUPER,
    /** Family-specific features encoding the exact decision-relevant state used by SuperDFS — see {@link SuperCustomFeatures}. */
    SUPER_CUSTOM,
    /** SUPER features plus cross-candidate index ranking; family-agnostic, no hardcoded substates — see {@link SuperGeneralFeatures}. */
    SUPER_GENERAL;

    public FeatureCompute create() {
        return switch (this) {
            case BASIC        -> new BasicFeatures();
            case ROL          -> new RolFeatures();
            case SUPER        -> new SuperFeatures();
            case SUPER_CUSTOM -> new SuperCustomFeatures(
                System.getProperty("superdfs.family", ""),
                Integer.parseInt(System.getProperty("superdfs.n", "1")),
                Integer.parseInt(System.getProperty("superdfs.k", "1")));
            case SUPER_GENERAL -> new SuperGeneralFeatures();
        };
    }
}
