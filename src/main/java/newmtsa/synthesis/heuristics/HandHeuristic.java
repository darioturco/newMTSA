package newmtsa.synthesis.heuristics;

import newmtsa.synthesis.ExtendedTransition;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Hand-crafted heuristics for directed controller synthesis.
 *
 * <p>Each family uses an internal finite state machine (variable {@code state}) to guide
 * which transition is expanded next. The states and transitions for each family:
 *
 * <pre>
 * ─────────────────────────────────────────────────────────────────────
 *  TL – Transfer Line
 * ─────────────────────────────────────────────────────────────────────
 *  States:
 *    noncontrolable, put.1, get.1, put.2, get.2, ..., put.n, get.n,
 *    returning, reject, return.get, accept, accept.get
 *
 *  Transitions:
 *    noncontrolable   --(get[0])-----------> put.1
 *    put.i            --(put[i])-----------> get.i        i = 1..n
 *    get.i            --(get[i])-----------> put.(i+1)    i = 1..n-1
 *    get.n            --(get[n])-----------> returning
 *    returning        --(return*)----------> reject
 *    reject           --(reject)-----------> return.get
 *    return.get       --(get[1])-----------> accept
 *    accept           --(accept)-----------> accept.get
 *    accept.get       --(get[0])-----------> (terminal)
 *
 * ─────────────────────────────────────────────────────────────────────
 *  DP – Dining Philosophers
 * ─────────────────────────────────────────────────────────────────────
 *  States:
 *    noncontrolable,
 *    step.0, eat.0, feeding.1, step.1, eat.1, ...,
 *    feeding.(n-1), step.(n-1), eat.(n-1),
 *    eat.all.first, eat.all.comeback, eat.all.second,
 *    eat.all.think, eat.all.third
 *
 *  Transitions:
 *    noncontrolable    --(uncontrollable*)---------> noncontrolable    (self-loop)
 *    noncontrolable    --(take[0][0], all Hungry)--> step.0
 *    step.i            --(step[i])----------------> step.i            (self-loop)
 *    step.i            --(take[i][(i+1)%n])-------> eat.i
 *    eat.i             --(eat[i]*, release[i]*)---> eat.i             (self-loop)
 *    eat.i             --(think[i])---------------> feeding.(i+1)     i < n-1
 *    eat.(n-1)         --(think[n-1])-------------> eat.all.first
 *    feeding.i         --(take[i][i], fresh)------> step.i
 *    eat.all.first     --(eat.all)----------------> eat.all.comeback
 *    eat.all.comeback  --(take[0][0])-------------> eat.all.second
 *    eat.all.second    --(eat.all)----------------> eat.all.think
 *    eat.all.think     --(think*)-----------------> eat.all.third
 *    eat.all.third     --(release*, eat.all)------> eat.all.third     (self-loop / terminal)
 *
 * ─────────────────────────────────────────────────────────────────────
 *  CM – Cat and Mouse
 * ─────────────────────────────────────────────────────────────────────
 *  States:
 *    noncontrolable,
 *    mouse.turn.0, mouse.turn.1, ..., mouse.turn.(n-1),
 *    cat.turn
 *
 *  Transitions:
 *    noncontrolable     --(uncontrollable)-----------> noncontrolable       (self-loop)
 *    noncontrolable     --(no uncontrollable left)---> mouse.turn.0
 *    mouse.turn.i       --(mouse[i].move[p])---------> mouse.turn.(i+1)    i < n-1
 *                          p = argmin|p - safePlace|, safePlace = k, fresh preferred
 *    mouse.turn.(n-1)   --(mouse[n-1].move[p])-------> cat.turn
 *    cat.turn           --(uncontrollable*)-----------> cat.turn            (self-loop)
 *    cat.turn           --(mouse.turn)---------------> mouse.turn.0
 *
 * ─────────────────────────────────────────────────────────────────────
 *  AT – Air Traffic  (studied for n = k)
 * ─────────────────────────────────────────────────────────────────────
 *  States:
 *    noncontrolable,
 *    descend.p  (p = 0..n-1, specific plane),
 *    descend.all,
 *    approach.i (i = plane currently at height 0),
 *    control.all
 *
 *  Transitions:
 *    noncontrolable  --(init, p=0)--------------------> descend.p
 *    descend.p       --(descend[p][h], !allHeight)-----> descend.nextFree   nextFree != -1
 *    descend.p       --(descend[p][h], !allHeight)-----> control.all        nextFree == -1
 *    descend.p       --(descend[p][h],  allHeight)-----> approach.X         !expanded
 *    descend.p       --(descend[p][h],  allHeight)-----> control.all        expanded
 *    descend.p       --(requestLand[p], fresh)---------> descend.p          no state change
 *    descend.p       --(stale frontier)----------------> control.all
 *    descend.all     --(descend[*][h], !expanded)------> descend.all        (self-loop)
 *    descend.all     --(descend[*][h],  expanded)------> control.all
 *    descend.all     --(no descend, planeAt0=i)--------> approach.i
 *    approach.i      --(approach[i], fresh)------------> approach.i         no state change
 *    approach.i      --(land[i], allLanded|expanded)---> control.all
 *    approach.i      --(land[i], !allLanded,!expanded)--> descend.all
 *    control.all     --(control.all action)------------> descend.1
 *    control.all     --(requestLand[i], fresh)---------> descend.i
 *    control.all     --(extendFlight[i], off,freePlane)> descend.freePlane
 *    control.all     --(extendFlight[i], off,expanded)-> control.all        (self-loop)
 *    control.all     --(extendFlight[i], off,planeAt0)-> approach.planeAt0
 *    control.all     --(requestLand[i], off,!expanded)-> descend.i
 * </pre>
 */
public class HandHeuristic implements Heuristic {

    private final String family;
    private final int n;
    private final int k;
    private SynthesisContext ctx;
    private boolean verbose;
    private String state = "noncontrolable";
    private Set<String> controllable;
    private int currentStep;
    private final int safePlace; // Only for CM
    private final Set<String> expandedStates = new HashSet<>();

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
        int idx = switch (family) {
            case "DP" -> dpPick(pending);
            case "TA" -> taPick(pending);
            case "AT" -> atPick(pending);
            case "TL" -> tlPick(pending);
            case "BW" -> bwPick(pending);
            case "CM" -> cmPick(pending);
            default  -> pending.size() - 1;
        };
        expandedStates.add(pending.get(idx).to());
        return idx;
    }

    private boolean isExpanded(String state) {
        return expandedStates.contains(state);
    }

    private int dpPick(List<ExtendedTransition> pending) {
        int i = extractIndex();
        String baseState = i != -1 ? state.substring(0, state.lastIndexOf(".")) : state;
        switch (baseState) {
            case "noncontrolable":
                for (int j = 0; j < pending.size(); j++) {
                    if (!controllable.contains(pending.get(j).action())) {
                        return j;
                    }
                }
                for (int j = 0; j < pending.size(); j++) {
                    ExtendedTransition t = pending.get(j);
                    if ("take[0][0]".equals(t.action()) && countHungry(t.from()) == n) {
                        setState("step.0");
                        return j;
                    }
                }
                break;
            case "feeding":
                String targetAction = "take[" + i + "][" + i + "]";
                for (int j = 0; j < pending.size(); j++) {
                    ExtendedTransition t = pending.get(j);
                    if (targetAction.equals(t.action()) && t.step() == currentStep - 1) {
                        setState("step." + i);
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
                        setState("eat." + i);
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
                        setState((i == n - 1) ? "eat.all.first" : "feeding." + (i + 1));
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
        int i = extractIndex();
        boolean newFrontier = pending.get(pending.size() - 1).step() == currentStep - 1;
        String baseState = i != -1 ? state.substring(0, state.lastIndexOf(".")) : state;
        if ("descend".equals(baseState) && !newFrontier) {
            setState("control.all");
            baseState = "control.all";
        }
        if (currentStep == 81){
            System.out.println("------------------------- Step Break -------------------------------");
        }
        switch (baseState) {
            case "noncontrolable":
                setState("descend.0");
                i = 0;
                // fall through
            case "descend":
                int bestIdx = findLowestDescend(pending, i);
                if (bestIdx != -1) {
                    ExtendedTransition t = pending.get(bestIdx);
                    if (allPlanesHaveHeight(t.to())) {
                        if (isExpanded(t.to())) {
                            setState("control.all");
                        } else {
                            setState("approach." + planeAtHeight0(t.to()));
                        }
                    } else {
                        int nextFree = findNextFreePlane(t.to());
                        if(nextFree == -1){
                            setState("control.all");
                        }else {
                            setState("descend." + nextFree);
                        }
                    }
                    return bestIdx;
                }
                String requestTarget = "requestLand[" + i + "]";
                for (int j = 0; j < pending.size(); j++) {
                    if (requestTarget.equals(pending.get(j).action()) && pending.get(j).step() == currentStep - 1) {
                        return j;
                    }
                }
                break;
            case "descend.all":
                // Busca acciones 'descend[i][a]' donde el from tenga Occupied[i]|Empty
                int pickIdx = findDescendWithEmptyBelow(pending);
                if (pickIdx != -1) {
                    if (isExpanded(pending.get(pickIdx).to())) {
                        setState("control.all");
                    }
                    return pickIdx;
                }
                // No hay descensos disponibles → pasamos a approach con el avion en altura 0
                int planeAt0 = planeAtHeight0(pending);
                if (planeAt0 != -1) {
                    i = planeAt0;
                    setState("approach." + i);
                }
                // fall through
            case "approach":
                String landTarget = "land[" + i + "]";
                for (int j = 0; j < pending.size(); j++) {
                    ExtendedTransition t = pending.get(j);
                    if (landTarget.equals(t.action()) && t.step() == currentStep - 1) {
                        if(countEnd(t.to()) == n){
                            setState("control.all");
                        }else if(isExpanded(t.to())){
                            setState("control.all");
                        }else{
                            setState("descend.all");
                        }
                        return j;
                    }
                }
                String approachTarget = "approach[" + i + "]";
                for (int j = 0; j < pending.size(); j++) {
                    if (approachTarget.equals(pending.get(j).action()) && pending.get(j).step() == currentStep - 1) {
                        return j;
                    }
                }
                break;
            case "control.all":
                for (int j = 0; j < pending.size(); j++) {
                    if ("control.all".equals(pending.get(j).action())) {
                        setState("descend.1");
                        return j;
                    }
                }
                if (newFrontier) {
                    // Busca el requestLand que se agrego en la ultima expansion
                    int rlIdx = findMinRequestLand(pending);
                    if (rlIdx != -1) {
                        setState("descend." + extractIndex(pending.get(rlIdx).action()));
                        return rlIdx;
                    }
                }else{
                    // Busca el proximo extendFlight a expandir, el cual es el que menos tiempo estubo en la frontera y no tenga el fluen on
                    for (int j = pending.size() - 1; j >= 0; j--) {
                        ExtendedTransition t = pending.get(j);
                        if (t.action().startsWith("extendFlight") && hasOff(t.from())) {
                            int freePlane = findNextFreePlane(t.to());
                            if (freePlane != -1 && !isExpanded(t.to())) {
                                setState("descend." + freePlane);
                            } else if (isExpanded(t.to())) {
                                setState("control.all");
                            } else {
                                setState("approach." + planeAtHeight0(t.to()));
                            }
                            return j;
                        }
                    }
                    for (int j = pending.size() - 1; j >= 0; j--) {
                        ExtendedTransition t = pending.get(j);
                        if (t.action().startsWith("requestLand") && hasOff(t.from())) {
                            int plane = extractIndex(t.action());
                            if (!isExpanded(t.to())) {
                                setState("descend." + plane);
                            }
                            return j;
                        }
                    }
                }


                break;
        }
        return pending.size() - 1;
    }

    private int tlPick(List<ExtendedTransition> pending) {
        int i = extractIndex();
        String baseState = i != -1 ? state.substring(0, state.lastIndexOf(".")) : state;
        switch (baseState) {
            case "noncontrolable":
                setState("get.0");
                // fall through
            case "get":
                for (int j = pending.size() - 1; j >= 0; j--) {
                    ExtendedTransition t = pending.get(j);
                    if (("get[" + i + "]").equals(t.action()) && (i == 0 || t.step() == currentStep - 1)) {
                        setState(i == n ? "returning" : "put." + (i + 1));
                        return j;
                    }
                }
                break;
            case "put":
                for (int j = pending.size() - 1; j >= 0; j--) {
                    if (("put[" + i + "]").equals(pending.get(j).action())) {
                        setState("get." + i);
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
        int i = extractIndex();
        String baseState = i != -1 ? state.substring(0, state.lastIndexOf(".")) : state;
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
                        if (r != k && n > 1) {
                            setState("recover." + i);
                        }
                        return j;
                    }
                }
                if (i != -1) {
                    String acceptAct = "accept[" + i + "]";
                    for (int j = pending.size() - 1; j >= 0; j--) {
                        ExtendedTransition t = pending.get(j);
                        if (acceptAct.equals(t.action()) && !t.to().contains("ERROR")) {
                            return j;
                        }
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
        int i = extractIndex();
        String baseState = i != -1 ? state.substring(0, state.lastIndexOf(".")) : state;

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

    private long countEnd(String compositeState) {
        long count = 0;
        for (String part : compositeState.split("\\|")) {
            if (part.equals("End")) count++;
        }
        return count;
    }

    private boolean hasOff(String compositeState) {
        String[] parts = compositeState.split("\\|");
        return parts.length > 0 && parts[parts.length - 1].contains("off");
    }

    private int extractIndex() {
        if (!state.contains(".")) return -1;
        String rest = state.substring(state.lastIndexOf(".") + 1);
        try {
            return Integer.parseInt(rest);
        } catch (NumberFormatException e) {
            return -1;
        }
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

    private int planeAtHeight0(List<ExtendedTransition> pending) {
        for (ExtendedTransition t : pending) {
            if (t.step() != currentStep - 1) continue;
            String[] parts = t.from().split("\\|");
            if (parts.length > 2) {
                String monitor0 = parts[2];
                if (monitor0.startsWith("Occupied[")) {
                    return Integer.parseInt(monitor0.substring(monitor0.indexOf('[') + 1, monitor0.indexOf(']')));
                }
            }
        }
        return -1;
    }

    private int findNextFreePlane(String toState) {
        String[] parts = toState.split("\\|");
        for (int i = 2 + k; i < 2 + k + n && i < parts.length; i++) {
            String part = parts[i];
            int hash = part.indexOf('#');
            if (hash != -1) part = part.substring(0, hash);
            if (part.startsWith("Airplane(") && part.endsWith(")")) {
                return Integer.parseInt(part.substring("Airplane(".length(), part.length() - 1));
            }
        }
        return -1;
    }

    private int findLowestDescend(List<ExtendedTransition> pending, int plane) {
        int bestP = Integer.MAX_VALUE;
        int bestIdx = -1;
        for (int j = 0; j < pending.size(); j++) {
            ExtendedTransition t = pending.get(j);
            String a = t.action();
            if (a.startsWith("descend[" + plane + "][") && t.step() == currentStep - 1) {
                int p = descendY(a);
                String[] parts = t.from().split("\\|");
                if (parts.length > 2 + p && "Empty".equals(parts[2 + p]) && p < bestP) {
                    bestP = p;
                    bestIdx = j;
                }
            }
        }
        return bestIdx;
    }

    private boolean allPlanesHaveHeight(String compositeState) {
        String[] parts = compositeState.split("\\|");
        // last part is fluent; n parts immediately before it are plane substates
        int fluent = parts.length - 1;
        for (int p = 1; p <= n; p++) {
            int idx = fluent - p;
            if (idx < 0) return false;
            String sub = parts[idx];
            int hash = sub.indexOf('#');
            if (hash != -1) sub = sub.substring(0, hash);
            if (!sub.startsWith("Holding[") && !sub.equals("End")) return false;
        }
        return true;
    }

    private int planeAtHeight0(String compositeState) {
        String[] parts = compositeState.split("\\|");
        if (parts.length > 2) {
            String monitor0 = parts[2];
            if (monitor0.startsWith("Occupied[")) {
                return Integer.parseInt(monitor0.substring(monitor0.indexOf('[') + 1, monitor0.indexOf(']')));
            }
        }
        return -1;
    }

    private int findDescendWithEmptyBelow(List<ExtendedTransition> pending) {
        // Los HeightMonitor comienzan en posicion 2 (tras ResponseMonitor y RampMonitor)
        // y estan ordenados de menor a mayor altura. Se busca Empty (inferior desocupada)
        // seguido de Occupied[p] (superior ocupada por el avion p).
        for (int j = 0; j < pending.size(); j++) {
            ExtendedTransition t = pending.get(j);
            String a = t.action();
            if (a.startsWith("descend[") && t.step() == currentStep - 1) {
                int plane = descendX(a);
                String[] parts = t.from().split("\\|");
                for (int p = 3; p < parts.length - 1; p++) {
                    if (parts[p].contains("Occupied[" + plane + "]") && "Empty".equals(parts[p - 1])) {
                        return j;
                    }
                }
            }
        }
        return -1;
    }

    private int findMinRequestLand(List<ExtendedTransition> pending) {
        int bestI = Integer.MAX_VALUE;
        int bestIdx = -1;
        for (int j = 0; j < pending.size(); j++) {
            ExtendedTransition t = pending.get(j);
            String a = t.action();
            if (a.startsWith("requestLand[") && t.step() == currentStep - 1) {
                int plane = extractIndex(a);
                if (plane < bestI) {
                    bestI = plane;
                    bestIdx = j;
                }
            }
        }
        return bestIdx;
    }

    private int extractIndex(String action) {
        int bracket1 = action.indexOf("[");
        int bracket2 = action.indexOf("]");
        return Integer.parseInt(action.substring(bracket1 + 1, bracket2));
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
