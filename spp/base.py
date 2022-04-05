import pydot
import numpy as np

from pydrake.geometry.optimization import (
    GraphOfConvexSets,
)
from pydrake.solvers.mathematicalprogram import (
    CommonSolverOption,
    SolverOptions,
)
from pydrake.solvers.gurobi import GurobiSolver
from pydrake.solvers.mosek import MosekSolver

from spp.preprocessing import removeRedundancies
from spp.rounding import (
    greedyForwardPathSearch,
)

class BaseSPP:
    def __init__(self, regions):
        self.names = None
        if type(regions) is dict:
            self.names = list(regions.keys())
            regions = list(regions.values())
        else:
            self.names = ["v" + str(ii) for ii in range(len(regions))]
        self.dimension = regions[0].ambient_dimension()
        self.regions = regions.copy()
        self.solver = None
        self.options = None
        self.rounding_fn = [greedyForwardPathSearch]
        for r in self.regions:
            assert r.ambient_dimension() == self.dimension

        self.spp = GraphOfConvexSets()
        self.graph_complete = True


    def findEdgesViaOverlaps(self):
        edges = []
        for ii in range(len(self.regions)):
            for jj in range(ii + 1, len(self.regions)):
                if self.regions[ii].IntersectsWith(self.regions[jj]):
                    edges.append((ii, jj))
                    edges.append((jj, ii))
        return edges

    def findStartGoalEdges(self, start, goal):
        edges = [[], []]
        for ii in range(len(self.regions)):
            if self.regions[ii].PointInSet(start):
                edges[0].append(ii)
            if self.regions[ii].PointInSet(goal):
                edges[1].append(ii)
        return edges

    def setSolver(self, solver):
        self.solver = solver

    def setSolverOptions(self, options):
        self.options = options

    def setPaperSolverOptions(self):
        self.options = SolverOptions()
        self.options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        self.options.SetOption(MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3)
        self.options.SetOption(MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1)
        self.options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_TOL_REL_GAP", 1e-3)
        self.options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_MAX_TIME", 3600.0)
        self.options.SetOption(GurobiSolver.id(), "MIPGap", 1e-3)
        self.options.SetOption(GurobiSolver.id(), "TimeLimit", 3600.0)

    def setRoundingStrategy(self, rounding_fn):
        if callable(rounding_fn):
            self.rounding_fn = [rounding_fn]
        elif isinstance(rounding_fn, list):
            assert len(rounding_fn) > 0
            for fn in rounding_fn:
                assert callable(fn)
            self.rounding_fn = rounding_fn
        else:
            raise ValueError("Rounding strategy must either be "
                             "a function or list of functions.")

    def ResetGraph(self, vertices):
        for v in vertices:
            self.spp.RemoveVertex(v)
        for edge in self.spp.Edges():
            edge.ClearPhiConstraints()

    def VisualizeGraph(self, file_type="svg"):
        graphviz = self.spp.GetGraphvizString(None, False)
        data = pydot.graph_from_dot_data(graphviz)[0]
        if file_type == "svg":
            return data.create_svg()
        elif file_type == "png":
            return data.create_png()
        else:
            raise ValueError("Unrecognized file type:", file_type)


    def solveSPP(self, start, goal, rounding, preprocessing, verbose):
        if not self.graph_complete:
            raise NotImplementedError(
                "Replanning on a graph that has undergone preprocessing is "
                "not supported yet. Please construct a new planner.")

        results_dict = {}
        if preprocessing:
            results_dict["preprocessing_stats"] = removeRedundancies(self.spp, start, goal, verbose=verbose)
            self.graph_complete = False

        result = self.spp.SolveShortestPath(start, goal, rounding, self.solver, self.options)

        if rounding:
            results_dict["relaxation_result"] = result
            results_dict["relaxation_solver_time"] = result.get_solver_details().optimizer_time
        else:
            results_dict["mip_result"] = result
            results_dict["mip_solver_time"] = result.get_solver_details().optimizer_time

        if not result.is_success():
            print("First solve failed")
            return None, None, results_dict

        if verbose:
            print("Solution\t",
                  "Success:", result.get_solution_result(),
                  "Cost:", result.get_optimal_cost(),
                  "Solver time:", result.get_solver_details().optimizer_time)

        # Extract path
        active_edges = []
        found_path = False
        for fn in self.rounding_fn:
            rounded_edges = fn(self.spp, result, start, goal)
            if rounded_edges is None:
                print(fn.__name__, "could not find a path.")
            else:
                found_path = True
            active_edges.append(rounded_edges)
        results_dict["rounded_paths"] = active_edges
        if not found_path:
            print("All rounding strategies failed to find a path.")
            return None, None, results_dict

        # Solve with hard edge choices
        if rounding:
            rounded_results = []
            best_cost = np.inf
            best_path = None
            best_result = None
            max_rounded_solver_time = 0.0
            for path_edges in active_edges:
                if path_edges is None:
                    rounded_results.append(None)
                    continue
                for edge in self.spp.Edges():
                    if edge in path_edges:
                        edge.AddPhiConstraint(True)
                    else:
                        edge.AddPhiConstraint(False)
                rounded_results.append(self.spp.SolveShortestPath(
                    start, goal, rounding, self.solver, self.options))
                max_rounded_solver_time = np.maximum(
                    rounded_results[-1].get_solver_details().optimizer_time,
                    max_rounded_solver_time)
                if (rounded_results[-1].is_success()
                    and rounded_results[-1].get_optimal_cost() < best_cost):
                    best_cost = rounded_results[-1].get_optimal_cost()
                    best_path = path_edges
                    best_result = rounded_results[-1]

            results_dict["best_path"] = best_path
            results_dict["best_result"] = best_result
            results_dict["rounded_results"] = rounded_results
            results_dict["max_rounded_solver_time"] =  max_rounded_solver_time

            if verbose:
                print("Rounded Solutions:")
                for r in rounded_results:
                    if r is None:
                        print("\t\tNo path to solve")
                        continue
                    print("\t\t",
                        "Success:", r.get_solution_result(),
                        "Cost:", r.get_optimal_cost(),
                        "Solver time:", r.get_solver_details().optimizer_time)

            if best_path is None:
                print("Second solve failed on all paths.")
                return best_path, best_result, results_dict
        else:
            best_path = [active_edges[0]]
            results_dict["best_path"] = best_path
            results_dict["best_result"] = result
            results_dict["mip_path"] = best_path

        if verbose:
            for edge in best_path:
                print("Added", edge.name(), "to path.")

        return best_path, best_result, results_dict
