package newmtsa.synthesis;

/** A transition in the extended state space: from → action → to. */
public record ExtendedTransition(String from, String action, String to) {}
