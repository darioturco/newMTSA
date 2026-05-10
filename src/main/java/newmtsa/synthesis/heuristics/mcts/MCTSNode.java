package newmtsa.synthesis.heuristics.mcts;

import newmtsa.synthesis.ExtendedTransition;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A node in the MCTS virtual search tree.
 *
 * <p>Each node stores:
 * <ul>
 *   <li>{@code priors} — P(a) from softmax of the network's raw scores at this node's frontier.</li>
 *   <li>{@code Q[a]} — running mean of backpropagated values through action a (Welford update).</li>
 *   <li>{@code N[a]} — visit count through action a.</li>
 * </ul>
 *
 * <p>Action selection uses AlphaZero's PUCT formula:
 * <pre>UCB(a) = Q[a] + cPuct * P[a] * sqrt(totalN) / (1 + N[a])</pre>
 */
class MCTSNode {

    final List<ExtendedTransition> frontier;
    final float[]                  priors;
    final double[]                 Q;
    final int[]                    N;
    int                            totalN = 0;
    final Map<Integer, MCTSNode>   children = new HashMap<>();

    private MCTSNode(List<ExtendedTransition> frontier, float[] priors) {
        this.frontier = frontier;
        this.priors   = priors;
        this.Q        = new double[frontier.size()];
        this.N        = new int[frontier.size()];
    }

    static MCTSNode fromScores(List<ExtendedTransition> frontier, float[] rawScores) {
        return new MCTSNode(frontier, softmax(rawScores));
    }

    /** PUCT selection: argmax over actions of Q[a] + cPuct * P[a] * sqrt(totalN) / (1 + N[a]). */
    int selectUCB(double cPuct) {
        double sqrtTotal = Math.sqrt(Math.max(1, totalN));
        int    best      = 0;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int a = 0; a < frontier.size(); a++) {
            double ucb = Q[a] + cPuct * priors[a] * sqrtTotal / (1.0 + N[a]);
            if (ucb > bestScore) { bestScore = ucb; best = a; }
        }
        return best;
    }

    /** Welford running-mean update after receiving {@code value} through {@code action}. */
    void update(int action, double value) {
        N[action]++;
        totalN++;
        Q[action] += (value - Q[action]) / N[action];
    }

    /** Index of the most-visited action (used to pick the real move after simulations). */
    int mostVisited() {
        int best = 0;
        for (int a = 1; a < N.length; a++) {
            if (N[a] > N[best]) best = a;
        }
        return best;
    }

    /** Numerically-stable softmax. */
    static float[] softmax(float[] scores) {
        float max = scores[0];
        for (float s : scores) if (s > max) max = s;
        float[] out = new float[scores.length];
        float   sum = 0f;
        for (int i = 0; i < scores.length; i++) {
            out[i] = (float) Math.exp(scores[i] - max);
            sum   += out[i];
        }
        if (sum > 0f) for (int i = 0; i < out.length; i++) out[i] /= sum;
        return out;
    }
}
