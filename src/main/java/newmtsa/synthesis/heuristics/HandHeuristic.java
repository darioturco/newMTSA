package newmtsa.synthesis.heuristics;

import newmtsa.synthesis.ExtendedTransition;
import java.util.List;
import java.util.Set;

public class HandHeuristic implements Heuristic {

    private final String family;
    private final int n;
    private final int k;
    private SynthesisContext ctx;
    private boolean verbose;
    private String state = "noncontrolable";
    private Set<String> controllable;
    private int currentStep;
    private int r = 0;
    private final int safePlace; // Only for CM

    public HandHeuristic(String family, int n, int k) {
        this.family = family;
        this.n = n;
        this.k = k;
        this.safePlace = (2 * k + 1) / 2;
    }

    @Override
    public void init(SynthesisContext ctx) {
        this.ctx = ctx;
        this.verbose = ctx.verbose();
        this.controllable = ctx.controllable();
    }

    private void setState(String newState) {
        if (verbose) System.out.println(state + " -> " + newState);
        state = newState;
    }

    @Override
    public int pick(List<ExtendedTransition> pending) {
        currentStep++;
        return switch (family) {
            case "DP" -> dpPick(pending);
            case "TA" -> taPick(pending);
            case "AT" -> atPick(pending);
            case "TL" -> tlPick(pending);
            case "BW" -> bwPick(pending);
            case "CM" -> cmPick(pending);
            default  -> pending.size() - 1;
        };
    }

    private int dpPick(List<ExtendedTransition> pending) {
        int i = state.contains("[") ? Integer.parseInt(state.substring(state.indexOf("[") + 1, state.indexOf("]"))) : 0;
        String baseState = state.contains("[") ? state.substring(0, state.indexOf("[")) : state;
        switch (baseState) {
            case "noncontrolable":
                for (int j = 0; j < pending.size(); j++) {
                    if (!controllable.contains(pending.get(j).action())) {
                        return j;
                    }
                }
                setState("init.exploration");
                // fall through
            case "init.exploration":
                for (int j = 0; j < pending.size(); j++) {
                    ExtendedTransition t = pending.get(j);
                    if ("take[0][0]".equals(t.action()) && countHungry(t.from()) == n) {
                        setState("step[0]");
                        return j;
                    }
                }
                break;
            case "feeding":
                String targetAction = "take[" + i + "][" + i + "]";
                for (int j = 0; j < pending.size(); j++) {
                    ExtendedTransition t = pending.get(j);
                    if (targetAction.equals(t.action()) && t.step() == currentStep - 1) {
                        setState("step[" + i + "]");
                        return j;
                    }
                }
                break;
            case "step":
                String stepAction = "step[" + i + "]";
                for (int j = 0; j < pending.size(); j++) {
                    if (stepAction.equals(pending.get(j).action())) {
                        return j;
                    }
                }
                String rightTake = "take[" + i + "][" + ((i + 1) % n) + "]";
                for (int j = 0; j < pending.size(); j++) {
                    if (rightTake.equals(pending.get(j).action())) {
                        setState("eat[" + i + "]");
                        return j;
                    }
                }
                break;
            case "eat":
                String eatPrefix = "eat[" + i + "]";
                for (int j = 0; j < pending.size(); j++) {
                    if (pending.get(j).action().startsWith(eatPrefix)) {
                        return j;
                    }
                }
                String releasePrefix = "release[" + i + "]";
                for (int j = 0; j < pending.size(); j++) {
                    if (pending.get(j).action().startsWith(releasePrefix)) {
                        return j;
                    }
                }
                String thinkAction = "think[" + i + "]";
                for (int j = 0; j < pending.size(); j++) {
                    if (thinkAction.equals(pending.get(j).action())) {
                        setState((i == n - 1) ? "eat.all.first" : "feeding[" + (i + 1) + "]");
                        return j;
                    }
                }
                break;
            case "eat.all.first":
                for (int j = pending.size() - 1; j >= 0; j--) {
                    if ("eat.all".equals(pending.get(j).action())) {
                        setState("eat.all.comeback");
                        return j;
                    }
                }
                break;
            case "eat.all.comeback":
                for (int j = pending.size() - 1; j >= 0; j--) {
                    if ("take[0][0]".equals(pending.get(j).action())) {
                        setState("eat.all.second");
                        return j;
                    }
                }
                break;
            case "eat.all.second":
                for (int j = pending.size() - 1; j >= 0; j--) {
                    if ("eat.all".equals(pending.get(j).action())) {
                        setState("eat.all.think");
                        return j;
                    }
                }
                break;
            case "eat.all.think":
                for (int j = pending.size() - 1; j >= 0; j--) {
                    if (pending.get(j).action().startsWith("think")) {
                        setState("eat.all.third");
                        return j;
                    }
                }
                break;
            case "eat.all.third":
                for (int j = 0; j < pending.size(); j++) {
                    if (pending.get(j).action().startsWith("release")) {
                        return j;
                    }
                }
                for (int j = 0; j < pending.size(); j++) {
                    if ("eat.all".equals(pending.get(j).action())) {
                        return j;
                    }
                }
                System.err.println("ERROR: no release or eat.all action in frontier");
                break;
        }
        return pending.size() - 1;
    }

    private int taPick(List<ExtendedTransition> pending) {
        return pending.size() - 1;
    }

    // Esto es extremadamente dificil. No parece ser posible de resolcer si guardar el estado de cada avion en cada estado de la planta (PARA, eso lo tenes en los subestados!!!)
    private int atPick(List<ExtendedTransition> pending) {
        int i = state.contains(".") ? Integer.parseInt(state.substring(state.indexOf(".") + 1)) : 0;
        String baseState = state.contains(".") ? state.substring(0, state.indexOf(".")) : state;
        switch (baseState) {
            case "noncontrolable":
                setState("descend.0");
                i = 0;
                r = 0;
                // fall through
            case "descend":
                String descendTarget = "descend[" + i + "][" + r + "]";
                for (int j = 0; j < pending.size(); j++) {
                    ExtendedTransition t = pending.get(j);
                    if (descendTarget.equals(t.action()) && t.step() == currentStep - 1) {
                        r = r + 1;
                        if (i >= n - 1) {
                            setState("descend.all");
                        } else {
                            setState("descend." + (i + 1));
                        }
                        return j;
                    }
                }
                String requestTarget = "requestLand[" + i + "]";
                for (int j = 0; j < pending.size(); j++) {
                    if (requestTarget.equals(pending.get(j).action())) {
                        return j;
                    }
                }
                break;
            case "descend.all":
                int pickIdx = findMaxDescend(pending);
                if (pickIdx != -1) {
                    if (descendY(pending.get(pickIdx).action()) == 0) {
                        setState("approach." + descendX(pending.get(pickIdx).action()));
                    }
                    return pickIdx;
                }
                break;
            case "approach":
                String landTarget = "land[" + i + "]";
                for (int j = 0; j < pending.size(); j++) {
                    ExtendedTransition t = pending.get(j);
                    if (landTarget.equals(t.action()) && t.step() == currentStep - 1) {
                        setState("descend.all");
                        return j;
                    }
                }
                String approachTarget = "approach[" + i + "]";
                for (int j = 0; j < pending.size(); j++) {
                    if (approachTarget.equals(pending.get(j).action())) {
                        return j;
                    }
                }
                break;
        }
        return pending.size() - 1;
    }

    private int tlPick(List<ExtendedTransition> pending) {
        int i = state.contains("[") ? Integer.parseInt(state.substring(state.indexOf("[") + 1, state.indexOf("]"))) : 0;
        String baseState = state.contains("[") ? state.substring(0, state.indexOf("[")) : state;
        switch (baseState) {
            case "noncontrolable":
                setState("get[0]");
                // fall through
            case "get":
                for (int j = pending.size() - 1; j >= 0; j--) {
                    ExtendedTransition t = pending.get(j);
                    if (("get[" + i + "]").equals(t.action()) && (i == 0 || t.step() == currentStep - 1)) {
                        setState(i == n ? "returning" : "put[" + (i + 1) + "]");
                        return j;
                    }
                }
                break;
            case "put":
                for (int j = pending.size() - 1; j >= 0; j--) {
                    if (("put[" + i + "]").equals(pending.get(j).action())) {
                        setState("get[" + i + "]");
                        return j;
                    }
                }
                break;
            case "returning":
                for (int j = pending.size() - 1; j >= 0; j--) {
                    if (pending.get(j).action().startsWith("return")) {
                        setState("reject");
                        return j;
                    }
                }
                break;
            case "reject":
                for (int j = pending.size() - 1; j >= 0; j--) {
                    if ("reject".equals(pending.get(j).action())) {
                        setState("return.get");
                        return j;
                    }
                }
                break;
            case "return.get":
                for (int j = pending.size() - 1; j >= 0; j--) {
                    if ("get[1]".equals(pending.get(j).action())) {
                        setState("accept");
                        return j;
                    }
                }
                break;
            case "accept":
                for (int j = pending.size() - 1; j >= 0; j--) {
                    if ("accept".equals(pending.get(j).action())) {
                        setState("accept.get");
                        return j;
                    }
                }
                break;
            case "accept.get":
                for (int j = pending.size() - 1; j >= 0; j--) {
                    if ("get[0]".equals(pending.get(j).action())) {
                        return j;
                    }
                }
                break;
        }
        return pending.size() - 1;
    }

    // No funciona para BW-3-3
    private int bwPick(List<ExtendedTransition> pending) {
        int i = state.contains(".") ? Integer.parseInt(state.substring(state.indexOf(".") + 1)) : 0;
        String baseState = state.contains(".") ? state.substring(0, state.indexOf(".")) : state;
        switch (baseState) {
            case "noncontrolable":
                setState("accept.0");
                i = 0;
                // fall through
            case "accept":
                for (int j = pending.size() - 1; j >= 0; j--) {
                    ExtendedTransition t = pending.get(j);
                    if ("approve".equals(t.action()) && !t.to().contains("ERROR")) {
                        setState("comeback");
                        return j;
                    }
                }
                String assignAction = "assign[" + i + "]";
                for (int j = pending.size() - 1; j >= 0; j--) {
                    ExtendedTransition t = pending.get(j);
                    if (assignAction.equals(t.action()) && t.step() == currentStep - 1) {
                        return j;
                    }
                }
                String acceptAction = "accept[" + i + "]";
                for (int j = pending.size() - 1; j >= 0; j--) {
                    if (acceptAction.equals(pending.get(j).action())) {
                        if (i != n) {
                            setState("accept." + (i + 1));
                        }
                        return j;
                    }
                }
                break;
            case "comeback":
                for (int j = pending.size() - 1; j >= 0; j--) {
                    ExtendedTransition t = pending.get(j);
                    if ("assign[0]".equals(t.action()) && t.step() == currentStep - 1) {
                        setState("reject." + (n - 1));
                        return j;
                    }
                }
                break;
            case "reject":
                for (int j = pending.size() - 1; j >= 0; j--) {
                    ExtendedTransition t = pending.get(j);
                    if ("refuse".equals(t.action()) && !t.to().contains("ERROR")) {
                        return j;
                    }
                }
                for (int j = pending.size() - 1; j >= 0; j--) {
                    String a = pending.get(j).action();
                    if (a.startsWith("reject[") && a.contains("][") && a.endsWith("]")) {
                        int r = secondIndexOfReject(a);
                        i = firstIndexOfReject(a);
                        if (r != k) {
                            setState("recover." + i);
                        }
                        return j;
                    }
                }
                break;
            case "recover":
                String assignRej = "assign[" + i + "]";
                for (int j = pending.size() - 1; j >= 0; j--) {
                    ExtendedTransition t = pending.get(j);
                    if (assignRej.equals(t.action()) && t.step() == currentStep - 1) {
                        return j;
                    }
                }
                String acceptRej = "accept[" + i + "]";
                for (int j = pending.size() - 1; j >= 0; j--) {
                    if (acceptRej.equals(pending.get(j).action())) {
                        setState("reject");
                        return j;
                    }
                }
                break;
        }
        return pending.size() - 1;
    }

    private int cmPick(List<ExtendedTransition> pending) {
        int i = 0;
        String baseState = state;
        if (state.contains(".")) {
            String lastPart = state.substring(state.lastIndexOf(".") + 1);
            try {
                i = Integer.parseInt(lastPart);
                baseState = state.substring(0, state.lastIndexOf("."));
            } catch (NumberFormatException e) {
                // keep full state as base (e.g. "cat.turn")
            }
        }

        switch (baseState) {
            case "noncontrolable":
                for (int j = 0; j < pending.size(); j++) {
                    if (!controllable.contains(pending.get(j).action())) {
                        return j;
                    }
                }
                setState("mouse.turn.0");
                i = 0;
                // fall through
            case "mouse.turn":
                int bestIdx = findBestMouseMove(pending, i);
                if (bestIdx != -1) {
                    if (i + 1 >= n) {
                        setState("cat.turn");
                    } else {
                        setState("mouse.turn." + (i + 1));
                    }
                    return bestIdx;
                }
                break;
            case "cat.turn":
                for (int j = 0; j < pending.size(); j++) {
                    String action = pending.get(j).action();
                    if (!controllable.contains(action) && !"mouse.turn".equals(action)) {
                        return j;
                    }
                }
                for (int j = 0; j < pending.size(); j++) {
                    if ("mouse.turn".equals(pending.get(j).action())) {
                        setState("mouse.turn.0");
                        return j;
                    }
                }
                break;
        }
        return pending.size() - 1;
    }

    private long countHungry(String compositeState) {
        long count = 0;
        for (String part : compositeState.split("\\|")) {
            if (part.equals("Hungry")) count++;
        }
        return count;
    }

    private int firstIndexOfReject(String action) {
        int firstBracket = action.indexOf("[");
        int secondBracket = action.indexOf("]");
        return Integer.parseInt(action.substring(firstBracket + 1, secondBracket));
    }

    private int secondIndexOfReject(String action) {
        int secondBracket = action.indexOf("]");
        int thirdBracket = action.indexOf("[", secondBracket);
        int fourthBracket = action.indexOf("]", thirdBracket);
        return Integer.parseInt(action.substring(thirdBracket + 1, fourthBracket));
    }

    private int findMaxDescend(List<ExtendedTransition> pending) {
        int maxY = -1;
        int pickIdx = -1;
        for (int j = 0; j < pending.size(); j++) {
            ExtendedTransition t = pending.get(j);
            String a = t.action();
            if (a.startsWith("descend[") && a.contains("][") && t.step() == currentStep - 1) {
                int y = descendY(a);
                if (y > maxY) {
                    maxY = y;
                    pickIdx = j;
                }
            }
        }
        return pickIdx;
    }

    private int descendX(String action) {
        int bracket1 = action.indexOf("[");
        int bracket2 = action.indexOf("]");
        return Integer.parseInt(action.substring(bracket1 + 1, bracket2));
    }

    private int descendY(String action) {
        int bracket2 = action.indexOf("]");
        int bracket3 = action.indexOf("[", bracket2);
        int bracket4 = action.indexOf("]", bracket3);
        return Integer.parseInt(action.substring(bracket3 + 1, bracket4));
    }

    private int findBestMouseMove(List<ExtendedTransition> pending, int mouseIdx) {
        int bestIdx = -1;
        int bestDist = Integer.MAX_VALUE;
        boolean bestIsFresh = false;
        String targetPrefix = "mouse[" + mouseIdx + "].move[";
        for (int j = 0; j < pending.size(); j++) {
            ExtendedTransition t = pending.get(j);
            String a = t.action();
            if (a.startsWith(targetPrefix)) {
                int p = extractMovePosition(a);
                int dist = Math.abs(p - safePlace);
                boolean fresh = t.step() == currentStep - 1;
                if (bestIdx == -1 ||
                    (fresh && !bestIsFresh) ||
                    (fresh == bestIsFresh && dist < bestDist)) {
                    bestIdx = j;
                    bestDist = dist;
                    bestIsFresh = fresh;
                }
            }
        }
        return bestIdx;
    }

    private int extractMovePosition(String action) {
        int lastBracket = action.lastIndexOf("[");
        int closeBracket = action.lastIndexOf("]");
        return Integer.parseInt(action.substring(lastBracket + 1, closeBracket));
    }
}
