import sys
import jpype
import jpype.imports
from pathlib import Path


def _start_jvm():
    if jpype.isJVMStarted():
        return
    jar = str(Path(__file__).parent / "mtsa.jar")
    if "linux" in sys.platform:
        jpype.startJVM(classpath=[jar])
    else:
        jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", classpath=[jar])


_start_jvm()

from newmtsa.synthesis import DCSForPython  # noqa: E402


class DCSEnvironment:
    """
    RL environment wrapping the on-the-fly non-blocking DCS algorithm.

    Observation : the current frontier.
                  Without features: list of (from_state, action, to_state) tuples.
                  With features:    list of int[] binary feature vectors.
    Action      : integer index into the frontier list.
    Reward      : always -1 (minimise steps to termination).
    Terminal    : when exploration ends (realizable or not).

    Parameters
    ----------
    heuristic : str
        Name of a HeuristicType enum constant (default: "FIRST").
        Java resolves the string to the matching Heuristic implementation.
    feature_type : str or None
        Name of a FeatureType enum constant (e.g. "BASIC").
        Java resolves the string to the matching FeatureCompute implementation.
        When None, getFrontier() is used and observations are (from, action, to) tuples.

    Example
    -------
    >>> env = DCSEnvironment(feature_type="BASIC")
    >>> obs = env.reset("path/to/instance.fsp")   # list of int[] feature vectors
    """

    def __init__(self, heuristic: str = "FIRST", feature_type: str = None, verbose: bool = False):
        self._heuristic_name    = heuristic
        self._feature_type_name = feature_type
        self._dcs               = None
        self._frontier          = []
        self.is_finished        = False
        jpype.JClass("java.lang.System").setProperty("rl.verbose", "true" if verbose else "false")

    def set_verbose(self, verbose: bool):
        jpype.JClass("java.lang.System").setProperty("rl.verbose", "true" if verbose else "false")

    # ── public API ──────────────────────────────────────────────────────────

    def reset(self, fsp_path: str) -> list:
        """
        Parse *fsp_path*, build a fresh DCS instance, return the initial frontier.

        The observation format depends on whether a feature_type was given:
        - With features: list of list[int] (binary feature vectors, one per frontier element)
        - Without:       list of (from, action, to) string tuples
        """
        self._dcs = DCSForPython.fromPath(
            fsp_path,
            self._heuristic_name,
            self._feature_type_name or "",
        )
        self._frontier   = self._read_frontier()
        self.is_finished = bool(self._dcs.isExplorationEnded())
        return self._frontier

    def step(self, action: int) -> tuple:
        """
        Expand the frontier element at *action* index.

        Returns (next_frontier, reward=-1, done, info).
        """
        if self.is_finished:
            raise RuntimeError("Episode already ended — call reset() first")
        if not (0 <= action < len(self._frontier)):
            raise IndexError(f"action {action} out of range [0, {len(self._frontier)})")

        self._dcs.expand(action)
        self._frontier   = self._read_frontier()
        self.is_finished = bool(self._dcs.isExplorationEnded())

        ctrl_exp     = int(self._dcs.getLastCtrlExpanded())
        non_ctrl_exp = int(self._dcs.getLastNonCtrlExpanded())
        reward       = -2 * ctrl_exp - non_ctrl_exp

        terminal_bonus = None
        if self.is_finished:
            episode_expansions = int(self._dcs.getTransitionsExplored())
            director_trans     = int(self._dcs.getDirectorTransitions())
            terminal_bonus     = 10 * min(director_trans - episode_expansions + 1, 0)
            reward += terminal_bonus

        info = {
            "transitions_explored": int(self._dcs.getTransitionsExplored()),
            "expansions":           ctrl_exp + non_ctrl_exp,
            "ctrl_expanded":        ctrl_exp,
            "non_ctrl_expanded":    non_ctrl_exp,
            "states_explored":      int(self._dcs.getStatesExplored()),
            "realizable":           bool(self._dcs.isRealizable()) if self.is_finished else None,
            "director_transitions":  director_trans if self.is_finished else None,
            "terminal_bonus":       terminal_bonus,
        }
        return self._frontier, reward, self.is_finished, info

    @property
    def frontier(self) -> list:
        return self._frontier

    def action_space_size(self) -> int:
        return len(self._frontier)

    def uses_features(self) -> bool:
        return self._feature_type_name is not None

    @property
    def is_nonblocking(self) -> bool:
        """True when the loaded instance is a non-blocking synthesis problem (not GR(1))."""
        return bool(self._dcs.isNonBlocking()) if self._dcs is not None else None

    # ── internal ────────────────────────────────────────────────────────────

    def _read_frontier(self) -> list:
        if self._feature_type_name is not None:
            return [list(f) for f in self._dcs.getFrontierWithFeatures()]
        result = []
        for t in self._dcs.getFrontier():
            result.append((
                str(getattr(t, "from")()),
                str(t.action()),
                str(t.to()),
            ))
        return result
