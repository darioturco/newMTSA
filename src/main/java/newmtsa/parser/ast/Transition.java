package newmtsa.parser.ast;

/** A single labeled edge in an LTS: source state --action--> target state. */
public record Transition(String from, String action, String to) {}
