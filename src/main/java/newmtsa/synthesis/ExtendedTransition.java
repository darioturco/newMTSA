package newmtsa.synthesis;

import java.util.Objects;

/** A transition in the extended state space: from → action → to. */
public final class ExtendedTransition {

    private final String from;
    private final String action;
    private final String to;
    private int step;

    public ExtendedTransition(String from, String action, String to) {
        this.from = from;
        this.action = action;
        this.to = to;
    }

    public ExtendedTransition(String from, String action, String to, int step) {
        this.from = from;
        this.action = action;
        this.to = to;
        this.step = step;
    }

    public String from()      { return from; }
    public String action()    { return action; }
    public String to()        { return to; }
    public int    step()      { return step; }
    public void   setStep(int s) { this.step = s; }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ExtendedTransition that)) return false;
        return from.equals(that.from) && action.equals(that.action) && to.equals(that.to);
    }

    @Override
    public int hashCode() {
        return Objects.hash(from, action, to);
    }

    public String format() {
        return from + " --[" + action + "]--> " + to;
    }

    @Override
    public String toString() {
        return "ExtendedTransition[from=" + from + ", action=" + action + ", to=" + to + "]";
    }
}
