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
 * </pre>
 */
public enum FeatureType {

    BASIC,
    /** BasicFeatures extended with role-based component state encoding — see {@link RolFeatures}. */
    ROL;

    public FeatureCompute create() {
        return switch (this) {
            case BASIC -> new BasicFeatures();
            case ROL   -> new RolFeatures();
        };
    }
}
