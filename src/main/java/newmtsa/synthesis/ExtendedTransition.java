package newmtsa.synthesis;

/** A transition in the extended state space: from → action → to. */
// TODO: delete this class, already exist called Transition that is similar to this.
public record ExtendedTransition(String from, String action, String to) {}
