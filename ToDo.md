The instances of test are wrong. The Goal spect, must have a set of liveness propierties, this set is the guaranties to proof in the GR(1) problem. Also the Goal can have a set of assuption (like the Test36), if that set dont appear in the goal the assuptions are empty. Also make sure the inital state of the fluet is False or off.

---

## 1. Fix the RA Heuristic — Graph Propagation (Gap Function)

**Status:** Partially implemented. RA.E structure rule #2 and RA.G (goals as targets) are now correct,
but the core distance-estimation logic is still wrong.

**The problem:**

Our `RAHeuristic` computes, for each component LTS, the BFS distance from the component's current state
to its nearest marked state. The composite estimate is the sum of these per-component distances.
This is a *direct* estimate: it only asks "how far is each component from being marked right now?"

The reference implementation (`ReadyAbstraction.java` in the old MTSA) does something fundamentally
different. It builds a **causal action-dependency graph** over the actions that are "ready" (enabled)
at the current composite state, and propagates estimates through a fixpoint closure.

**How the old RA works:**

1. For every action `t` that is currently ready (frontier), the direct cost is the sum of per-component
   BFS distances from each component's post-`t` state to that component's marked states.

2. For every pair of actions `(l, t)` where `l` is a *local* action (not yet in the frontier) and `t`
   is a ready action: if `l` "enables" `t` — meaning `l` advances some component from a state where `t`
   is not locally available to a state where it is — then the RA graph gets an edge `l -> t` with weight
   `gap(l, t)`.

3. `gap(l, t)` = for each component `i`, max over all paths from the current state `s_i` via `l` to
   some state `s'_i` where `t` is locally ready, of the BFS distance from `s'_i` to `s_i`'s marked
   states. In other words: how much further from marked does component `i` get when we take `l` in
   order to enable `t`?

4. Estimates propagate: `estimate(l) = estimate(t) + gap(l, t)` for each such edge, and the fixpoint
   is computed by iterating until stable (Bellman-Ford style on the RA graph).

5. The final estimate for a frontier transition is `estimate(action)` from this closure, not the raw
   BFS distance sum.

**What to implement:**

- At each composite state, build the set of "ready actions" (actions that are enabled = in the
  frontier or locally enabled in all participating components).
- For each component LTS, precompute `manyStepsReachableStates(s)` (transitive closure of transitions
  from state `s`) — used to check whether action `t` can ever become ready from state `s` via `l`.
- Build the RA graph: vertices = actions (ready + potentially enabling), edges from the dependency
  analysis above.
- Run fixpoint propagation to get final estimates.
- Use these estimates in `compare()` instead of raw BFS sums.

**Reference file:** `E:\Code\Java\mtsa\maven-root\mtsa\src\main\java\MTSTools\ac\ic\doc\mtstools\model\operations\DCS\nonblocking\ReadyAbstraction.java`

---

## 2. Frontier-Matching Tests (Compare Expansions with Old Implementation)

**Status:** JAR traces collected for all 6 benchmark families (2-2 instances). Tests not yet written.

**Goal:** For each family, run our synthesis using a scripted heuristic that replays the same action
choices as the old implementation, then assert that the frontier (set of available transitions) at
each step matches in size and action multiset.

**Collected data (old JAR, 10 steps each):**

| Family | Expansion script (actions, dot-notation) | Frontier sizes |
|--------|------------------------------------------|----------------|
| TL-2-2 | get.0, put.1, get.1, put.2, get.2, return.1, accept, reject, get.1, get.0 | 3,4,6,9,11,14,16,18,20,19 |
| AT-2-2 | requestLand.0, requestLand.1, extendFlight.0, requestLand.1, extendFlight.1, requestLand.0, extendFlight.1, control.all, requestLand.0, requestLand.1 | 4,5,6,7,8,9,10,10,13,12 |
| BW-2-2 | assign.0, reject.0.1, accept.0, assign.0, accept.0, reject.0.2, assign.1, reject.1.1, accept.1, approve | 4,6,9,12,14,13,16,15,18,20 |
| CM-2-2 | mouse.turn, mouse.1.move.3, mouse.0.move.3, cat.turn, cat.0.move.0, cat.0.move.1, cat.1.move.0, cat.1.move.1, cat.1.move.0, cat.0.move.0 | 1,4,5,5,8,9,10,11,12,12 |
| DP-2-2 | think.0, think.1, think.1, think.0, take.0.0, step.0, step.0, take.0.1, eat.0, release.0.0 | 2,3,4,5,4,5,6,7,7,7 |
| TA-2-2 | agency.request, query.0, available.0, steps.0.0, query.succ.0, steps.0.1, query.succ.0, select.0, select.0, unavailable.0 | 1,3,2,3,3,3,3,3,2,3 |

**Note:** Old code uses dot-notation (`get.0`); our code uses bracket-notation (`get[0]`). Conversion:
split by `.`, numeric tokens -> `[N]`, join (e.g., `mouse.1.move.3` -> `mouse[1].move[3]`).

**Implementation plan:**

1. Create `ActionScriptedHeuristic`: receives a `List<String>` of action names; on each `choose()` call,
   picks the first frontier transition whose action matches the next script entry (bracket-notation),
   removes that entry from the queue.
2. Create `FamilyFrontierMatchTest` (one `@Test` per family or one parameterized class).
3. Instrument synthesis to capture frontier size after each expansion (before choosing the next one).
4. Assert `frontierSize[i] == expected[i]` and that the multiset of action labels in the frontier
   matches (or is a superset for steps with >10 entries, to keep tests stable across tie-breaking).
5. Fix any bugs discovered.

---

## 3. Make the Benchmark Match the Old Implementation

**Status:** Not started.

**Goal:** Run the full benchmark suite (all families, all sizes) and compare:
- Realizability verdict (must match exactly)
- States explored and transitions expanded (should match or be close)
- Wall-clock time (our implementation should be competitive)

**Steps:**

1. Use `Benchmark.java` to run all instances.
2. Run the old JAR on the same instances with the same families.
3. Compare CSV outputs side by side.
4. Investigate any verdict mismatches first (correctness), then count mismatches (efficiency).
5. After RA graph propagation is fixed (Task 1), re-run and check if counts converge.

---

## 4. Properly Implement DCS for GR(1)

**Status:** Current implementation has many known flaws.

**Goal:** Full, correct On-The-Fly Directed Controller Synthesis for GR(1) specifications.

**Known issues to address:**

- The GR(1) synthesis loop (alternating between guarantee and assumption goals) is not correctly
  integrated with the OTF exploration. States need to track which goal is currently being pursued.
- Losing/winning state detection for GR(1) differs from non-blocking: a state is losing if the
  environment can force a cycle that avoids all guarantees, not just if it is blocking.
- The heuristic needs to be adapted for the GR(1) goal structure (cycling through guarantee indices).
- Parallel composition with guarantee LTS components needs to handle acceptance conditions correctly.
- The reference implementation is in the old MTSA codebase -- use it as the specification for
  correct behavior.

**Reference:** `E:\Code\Java\mtsa\maven-root\mtsa\src\main\java\MTSTools\ac\ic\doc\mtstools\model\operations\DCS\` (GR(1) subdirectory)
