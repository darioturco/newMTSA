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

from java.nio.file import Paths           # noqa: E402
from java.util import ArrayList, HashSet  # noqa: E402
from newmtsa.parser import FSPParser                              # noqa: E402
from newmtsa.synthesis.nonblocking import DCSForPython            # noqa: E402
from newmtsa.synthesis.heuristics import HeuristicType            # noqa: E402
from newmtsa.synthesis.features import FeatureType                # noqa: E402


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
    heuristic : HeuristicType  (Java enum)
        Heuristic used internally by DCSForPython (default: FIRST).
    feature_type : str or None
        Name of a FeatureType enum constant (e.g. "BASIC").
        Java resolves the string to the matching FeatureCompute implementation.
        When None, getFrontier() is used and observations are (from, action, to) tuples.

    Example
    -------
    >>> env = DCSEnvironment(feature_type="BASIC")
    >>> obs = env.reset("path/to/instance.fsp")   # list of int[] feature vectors
    """

    def __init__(self, heuristic=None, feature_type: str = None):
        self._heuristic_type  = heuristic or HeuristicType.FIRST
        self._feature_type_name = feature_type   # plain Python string, e.g. "BASIC"
        self._dcs   = None
        self._frontier = []
        self._done  = False

    # ── public API ──────────────────────────────────────────────────────────

    def reset(self, fsp_path: str) -> list:
        """
        Parse *fsp_path*, build a fresh DCS instance, return the initial frontier.

        The observation format depends on whether a feature_type was given:
        - With features: list of list[int] (binary feature vectors, one per frontier element)
        - Without:       list of (from, action, to) string tuples
        """
        model = FSPParser.parse(Paths.get(fsp_path))

        spec = None
        for s in model.controllerSpecs():
            if s.nonblocking():
                spec = s
        if spec is None:
            raise ValueError(f"No nonblocking controller spec found in: {fsp_path}")

        components = ArrayList()
        for p in model.processes():
            components.add(p)

        safety_props = ArrayList()
        for name in spec.safety():
            for prop in model.ltlProperties():
                if str(prop.name()) == str(name):
                    safety_props.add(prop)
                    break

        marking = HashSet()
        for m in spec.marking():
            marking.add(str(m))

        controllable = HashSet()
        for c in spec.controllable():
            controllable.add(str(c))

        heuristic     = self._heuristic_type.create()
        instance_name = Path(fsp_path).stem

        if self._feature_type_name is not None:
            fc = FeatureType.valueOf(self._feature_type_name).create()
            self._dcs = DCSForPython(
                instance_name, components, safety_props, marking, controllable, heuristic, fc
            )
        else:
            self._dcs = DCSForPython(
                instance_name, components, safety_props, marking, controllable, heuristic
            )

        self._done     = bool(self._dcs.isExplorationEnded())
        self._frontier = self._read_frontier()
        return self._frontier

    def step(self, action: int) -> tuple:
        """
        Expand the frontier element at *action* index.

        Returns (next_frontier, reward=-1, done, info).
        """
        if self._done:
            raise RuntimeError("Episode already ended — call reset() first")
        if not (0 <= action < len(self._frontier)):
            raise IndexError(f"action {action} out of range [0, {len(self._frontier)})")

        self._dcs.expand(action)
        self._done     = bool(self._dcs.isExplorationEnded())
        self._frontier = self._read_frontier()

        info = {
            "transitions_explored": int(self._dcs.getTransitionsExplored()),
            "states_explored":      int(self._dcs.getStatesExplored()),
            "realizable": bool(self._dcs.isRealizable()) if self._done else None,
        }
        return self._frontier, -1, self._done, info

    @property
    def frontier(self) -> list:
        return self._frontier

    def action_space_size(self) -> int:
        return len(self._frontier)

    def uses_features(self) -> bool:
        return self._feature_type_name is not None

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
