package newmtsa.synthesis;

import java.util.Optional;

/** Result of GR(1) synthesis: either a {@link Director} or UNREALIZABLE, plus exploration stats. */
public final class SynthesisResult {

    public static final SynthesisResult UNREALIZABLE =
            new SynthesisResult(null, 0, 0);

    private final Director director;
    private final int statesExplored;
    private final int transitionsExplored;

    private SynthesisResult(Director d, int states, int transitions) {
        this.director            = d;
        this.statesExplored      = states;
        this.transitionsExplored = transitions;
    }

    public static SynthesisResult unrealizable(int states, int transitions) {
        return new SynthesisResult(null, states, transitions);
    }

    public static SynthesisResult of(Director d, int states, int transitions) {
        if (d == null) throw new NullPointerException("director");
        return new SynthesisResult(d, states, transitions);
    }

    public boolean isRealizable()          { return director != null; }
    public Optional<Director> director()   { return Optional.ofNullable(director); }
    public int statesExplored()            { return statesExplored; }
    public int transitionsExplored()       { return transitionsExplored; }

    @Override
    public String toString() {
        return director != null ? director.toString() : "UNREALIZABLE";
    }
}
