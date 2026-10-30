"""Build joint optimization matrices and solve them with Gurobi or COPT."""

import contextlib
import io
import subprocess
import sys
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
from scipy.sparse import coo_matrix
from config import env_params, RL_params

# Guarded backends; a missing package leaves its name as None.
try:
    import gurobipy as gp
except ImportError:  # pragma: no cover - Gurobi backend unavailable without gurobipy
    gp = None

try:
    import coptpy as cp
except ImportError:  # pragma: no cover - COPT backend unavailable without coptpy
    cp = None


_GUROBI_STATUS_NAMES = (
    "LOADED", "OPTIMAL", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED", "CUTOFF",
    "ITERATION_LIMIT", "NODE_LIMIT", "TIME_LIMIT", "SOLUTION_LIMIT",
    "INTERRUPTED", "NUMERIC", "SUBOPTIMAL", "INPROGRESS", "USER_OBJ_LIMIT",
)
_COPT_STATUS_NAMES = (
    "OPTIMAL", "INFEASIBLE", "UNBOUNDED", "INF_OR_UNB", "NUMERICAL",
    "NODELIMIT", "IMPRECISE", "TIMEOUT", "UNFINISHED", "INTERRUPTED",
)


def _status_name(namespace_getter, names, status):
    """Resolve a solver status code to its human-readable constant name."""
    try:
        namespace = namespace_getter()
    except AttributeError:
        return f"unknown status {status}"
    for name in names:
        value = getattr(namespace, name, None)
        if isinstance(value, int) and value == status:
            return name
    return f"unknown status {status}"


class Optimizer:
    """Gurobi/COPT front-end: constraint assembly, MIP solving, and the four seed streams."""
    def __init__(self):
        """Start an empty COO constraint accumulator."""
        self.rows, self.cols, self.vals = [], [], []   # COO triplets
        self.lows, self.highs = [], []                 # per-row lower/upper bounds

    @staticmethod
    @lru_cache(maxsize=None)
    def _gurobi_licensed():
        """True if Gurobi solves a probe without a restricted/size-limited banner."""
        if gp is None:
            return False
        model = None
        banner = io.StringIO()
        try:
            with contextlib.redirect_stdout(banner), \
                    contextlib.redirect_stderr(io.StringIO()):
                model = gp.Model("license_probe")
                model.Params.OutputFlag = 0
                model.Params.Threads = 1
                model.Params.TimeLimit = 1.0
                variable = model.addVar(
                    lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="x"
                )
                model.setObjective(variable, gp.GRB.MINIMIZE)
                model.optimize()
                if model.Status != gp.GRB.OPTIMAL:
                    return False
            lowered = banner.getvalue().lower()
            return not any(
                marker in lowered
                for marker in (
                    "restricted license",
                    "size-limited",
                    "size limited",
                    "non-production use",
                )
            )
        except Exception:
            return False
        finally:
            if model is not None:
                try:
                    model.dispose()
                except Exception:
                    pass

    # Subprocess probe: capture the COPT banner and reject non-commercial diagnostics.
    _COPT_PROBE = (
        "import coptpy as cp\n"
        "e = cp.Envr()\n"
        "m = e.createModel('license_probe')\n"
        "m.setParam(cp.COPT.Param.Logging, 0)\n"
        "m.setParam(cp.COPT.Param.Threads, 1)\n"
        "v = m.addVar(lb=0.0, ub=1.0, vtype=cp.COPT.CONTINUOUS, name='x')\n"
        "m.setObjective(cp.quicksum([v]), cp.COPT.MINIMIZE)\n"
        "m.solve()\n"
        "print('PROBE_OK', int(m.status == cp.COPT.OPTIMAL), flush=True)\n"
        "e.close()\n"
    )

    @staticmethod
    @lru_cache(maxsize=None)
    def _copt_licensed():
        """True if COPT solves a probe outside its size-limited non-commercial mode."""
        if cp is None:
            return False
        try:
            result = subprocess.run(
                [sys.executable, "-c", Optimizer._COPT_PROBE],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=60,
            )
        except Exception:
            return False
        output = f"{result.stdout}\n{result.stderr}"
        if result.returncode != 0 or "PROBE_OK 1" not in output:
            return False
        lowered = output.lower()
        return not any(
            marker in lowered
            for marker in ("no license found", "size limitations", "non-commercial")
        )

    @staticmethod
    def detect_solver(preferred="gurobi"):
        """Probe in preferred order and return the first licensed backend."""
        supported = ("gurobi", "copt")
        order = (preferred,) + tuple(name for name in supported if name != preferred)
        for name in order:
            licensed = (
                Optimizer._gurobi_licensed()
                if name == "gurobi"
                else Optimizer._copt_licensed()
            )
            if licensed:
                return name
        raise RuntimeError(
            "No licensed MILP backend is available: install gurobipy (Gurobi) or coptpy "
            "(COPT), activate a valid license, then rerun. "
            f"Probed backends: {', '.join(order)}."
        )

    @staticmethod
    def solver_time_limit():
        """Return the effective per-model wall-clock limit (COPT gets 1.5x the base)."""
        base = RL_params["solver_time_limit"]
        if base is None:
            return None
        return 1.5 * float(base) if RL_params["solver"] == "copt" else float(base)

    @staticmethod
    def exploration_time_limit():
        """Return the exploration quick-solve wall-clock cap (COPT gets 1.5x)."""
        base = RL_params.get("exploration_time_limit")
        if base is None:
            return Optimizer.solver_time_limit()
        return 1.5 * float(base) if RL_params["solver"] == "copt" else float(base)

    @staticmethod
    def configure_solver(preferred="gurobi"):
        """Write RL_params["solver"] once; the configured time base stays untouched."""
        RL_params["solver"] = Optimizer.detect_solver(preferred=preferred)
        return RL_params["solver"]

    @staticmethod
    def learning_seeds(seed=None):
        """Derive initialization/exploration/replay/solver int seeds from one root."""
        root = int(RL_params["seed"] if seed is None else seed)
        names = ("initialization", "exploration", "replay", "solver")
        children = np.random.SeedSequence(root).spawn(len(names))
        return {
            # Map each child's uint32 state to the positive range [0, 2^31-1).
            name: int(child.generate_state(1)[0]) % (2**31 - 1)
            for name, child in zip(names, children)
        }

    def add_constraint(self, coefficients, lo, hi):
        """Register lo <= sum(val*x) <= hi as sparse coefficients."""
        row = len(self.lows)
        for col, val in coefficients.items():
            if val:
                self.rows.append(row)
                self.cols.append(col)
                self.vals.append(val)
        self.lows.append(lo)
        self.highs.append(hi)

    def matrix(self, size):
        """Return the (K, size) CSC constraint matrix."""
        return coo_matrix(
            (self.vals, (self.rows, self.cols)), shape=(len(self.lows), size)
        ).tocsc()

    @staticmethod
    def _threads():
        """Return the solver thread count from the configuration."""
        threads = int(RL_params["solver_threads"])
        return threads

    @staticmethod
    def gurobi_solve(objective, integrality, lower, upper, matrix, lows, highs,
                     solver_seed=None, policy=None, quick=False):
        """Solve the bounded mixed-integer model with Gurobi and return its incumbent."""
        # matrix: float64 sparse (M,N); coefficients/bounds retain solver precision.
        if gp is None:
            raise ImportError(
                "gurobipy is required for solver='gurobi'; install it or select solver='copt'"
            )
        model = gp.Model("BMIP" if policy is None else f"BMIP_{policy}")
        model.Params.OutputFlag = 0
        time_limit = Optimizer.solver_time_limit()
        if time_limit is not None:
            model.Params.TimeLimit = time_limit
        if RL_params["solver_node_limit"] is not None:
            model.Params.NodeLimit = RL_params["solver_node_limit"]
        model.Params.Threads = Optimizer._threads()
        model.Params.MIPGap = RL_params["solver_mip_gap"]
        model.Params.FeasibilityTol = float(env_params["solver_tolerance"])
        model.Params.OptimalityTol = float(env_params["solver_tolerance"])
        model.Params.IntFeasTol = float(env_params["solver_tolerance"])
        # Feasibility focus: a usable incumbent beats the all-reject fallback at the time limit.
        model.Params.MIPFocus = 1
        model.Params.Seed = (
            Optimizer.learning_seeds()["solver"] if solver_seed is None else int(solver_seed)
        )
        if quick:
            # Exploration: short budget, best incumbent without proving optimality.
            quick_limit = Optimizer.exploration_time_limit()
            if quick_limit is not None:
                model.Params.TimeLimit = quick_limit

        # Integral positions are binary; the rest continuous.
        variables = model.addMVar(
            len(objective),
            lb=lower,
            ub=upper,
            vtype=np.where(integrality, gp.GRB.BINARY, gp.GRB.CONTINUOUS),
        )
        model.setObjective(objective @ variables, gp.GRB.MINIMIZE)
        # Equal bounds -> equality; otherwise split into lo/hi constraints.
        for i in range(len(lows)):
            expression = matrix.getrow(i) @ variables
            if lows[i] == highs[i]:
                model.addConstr(expression == lows[i])
            else:
                if np.isfinite(lows[i]):
                    model.addConstr(expression >= lows[i])
                if np.isfinite(highs[i]):
                    model.addConstr(expression <= highs[i])

        model.optimize()
        status = int(model.Status)
        status_name = _status_name(
            lambda: gp.GRB.Status, _GUROBI_STATUS_NAMES, status
        )
        result = SimpleNamespace(
            x=variables.X if model.SolCount else None,
            status=status,
            status_name=status_name,
            message=f"Gurobi {status_name} of policy {policy}",
            mip_gap=(model.MIPGap if model.IsMIP else 0.0) if model.SolCount else None,
            policy=policy,
        )
        model.dispose()
        return result

    @staticmethod
    def copt_solve(objective, integrality, lower, upper, matrix, lows, highs,
                   policy=None, quick=False):
        """Solve the same bounded mixed-integer model with COPT and return its incumbent."""
        # COPT exposes no seed parameter; fixed threads are recorded for reproducibility.
        if cp is None:
            raise ImportError(
                "coptpy is required for solver='copt'; install it or select solver='gurobi'"
            )

        # Suppress COPT's per-Env license banner.
        with contextlib.redirect_stdout(io.StringIO()):
            env = cp.Envr()
        model = env.createModel("BMIP" if policy is None else f"BMIP_{policy}")
        model.setParam(cp.COPT.Param.Logging, 0)
        model.setParam(cp.COPT.Param.Threads, Optimizer._threads())
        model.setParam(cp.COPT.Param.RelGap, RL_params["solver_mip_gap"])
        model.setParam(cp.COPT.Param.FeasTol, float(env_params["solver_tolerance"]))
        model.setParam(cp.COPT.Param.DualTol, float(env_params["solver_tolerance"]))
        model.setParam(cp.COPT.Param.IntTol, float(env_params["solver_tolerance"]))
        # No MIPFocus: raise primal heuristics for an early incumbent (each guarded).
        for name, value in (
            ("HeurLevel", 3),
            ("RoundingHeurLevel", 3),
            ("DivingHeurLevel", 3),
            ("FAPHeurLevel", 3),
        ):
            try:
                model.setParam(getattr(cp.COPT.Param, name), value)
            except Exception:  # pragma: no cover - parameter must never break the solve
                pass
        time_limit = Optimizer.solver_time_limit()
        if time_limit is not None:
            model.setParam(cp.COPT.Param.TimeLimit, time_limit)
        if quick:
            # Exploration: same short-budget best-incumbent contract as Gurobi.
            quick_limit = Optimizer.exploration_time_limit()
            if quick_limit is not None:
                model.setParam(cp.COPT.Param.TimeLimit, quick_limit)
        if RL_params["solver_node_limit"] is not None:
            model.setParam(cp.COPT.Param.NodeLimit, RL_params["solver_node_limit"])

        # Integral positions are binary; the rest continuous.
        variables = [
            model.addVar(
                lb=float(lower[j]),
                ub=float(upper[j]),
                vtype=cp.COPT.BINARY if integrality[j] else cp.COPT.CONTINUOUS,
            )
            for j in range(len(objective))
        ]
        model.setObjective(
            cp.quicksum(c * v for c, v in zip(objective, variables)), cp.COPT.MINIMIZE
        )

        # Equal bounds -> equality; otherwise split into lo/hi constraints.
        for i in range(len(lows)):
            row = matrix.getrow(i)
            expression = cp.quicksum(
                c * variables[j] for c, j in zip(row.data, row.indices)
            )
            if lows[i] == highs[i]:
                model.addConstr(expression == lows[i])
            else:
                if np.isfinite(lows[i]):
                    model.addConstr(expression >= lows[i])
                if np.isfinite(highs[i]):
                    model.addConstr(expression <= highs[i])

        model.solve()
        status = int(model.status)
        status_name = _status_name(lambda: cp.COPT, _COPT_STATUS_NAMES, status)
        # HasSol reports a usable incumbent; ObjVal stays the 1e+30 sentinel without one.
        if status == cp.COPT.OPTIMAL:
            has_solution = True
        else:
            try:
                has_solution = bool(model.getAttr(cp.COPT.Attr.HasSol))
            except Exception:  # pragma: no cover - attribute probe must never hide a solve
                has_solution = False
        x = np.array([v.x for v in variables], dtype=float) if has_solution else None

        mip_gap = None
        if has_solution:
            if np.any(integrality):
                try:
                    gap = float(model.BestGap)
                    # BestGap mirrors Gurobi MIPGap; the 1e+30 sentinel means unknown.
                    mip_gap = gap if np.isfinite(gap) and gap < 1e30 else None
                except (AttributeError, TypeError, ValueError):
                    mip_gap = None
            else:
                mip_gap = 0.0

        result = SimpleNamespace(
            x=x,
            status=status,
            status_name=status_name,
            message=f"COPT {status_name} of policy {policy}",
            mip_gap=mip_gap,
            policy=policy,
        )
        env.close()
        return result
