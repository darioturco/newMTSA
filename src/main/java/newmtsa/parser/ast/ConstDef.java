package newmtsa.parser.ast;

/**
 * Represents a constant declaration: {@code const N = 2}
 *
 * <p>The {@code value} is fully resolved to an integer at parse time.
 * References to other constants (e.g. {@code const Philosophers = N}) are
 * looked up and substituted; basic arithmetic (+, -, *, /, %) is evaluated.
 *
 * <p>The {@code name} is unique within an {@link FSPModel}: if the same
 * constant name appears more than once in the source, subsequent occurrences
 * are renamed by appending an underscore and a counter ({@code N_1}, {@code N_2}, …).
 */
public record ConstDef(String name, int value) {}
