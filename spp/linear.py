import numpy as np
import pydot
import time

from pydrake.geometry.optimization import (
    Point,
)
from pydrake.solvers.mathematicalprogram import (
    Binding,
    Constraint,
    Cost,
    L2NormCost,
    LinearConstraint,
)

from spp.base import BaseSPP

class LinearSPP(BaseSPP):
    def __init__(self, regions, edges=None, path_weights=None):
        BaseSPP.__init__(self, regions)

        if path_weights is None:
            path_weights = np.ones(self.dimension)
        assert len(path_weights) == self.dimension

        self.edge_cost = L2NormCost(
            np.hstack((np.diag(-path_weights), np.diag(path_weights))),
            np.zeros(self.dimension))

        for i, r in enumerate(self.regions):
            self.spp.AddVertex(r, name = self.names[i] if not self.names is None else '')

        if edges is None:
            edges = self.findEdgesViaOverlaps()

        vertices = self.spp.Vertices()
        for ii, jj in edges:
            u = vertices[ii]
            v = vertices[jj]
            edge = self.spp.AddEdge(u, v, f"({u.name()}, {v.name()})")

            edge_length = edge.AddCost(Binding[Cost](
                self.edge_cost, np.append(u.x(), v.x())))[1]
            self.edge_cost_dict[edge.id()] = [edge_length]

            # Constrain point in v to be in u
            edge.AddConstraint(Binding[Constraint](
                LinearConstraint(u.set().A(),
                                 -np.inf*np.ones(len(u.set().b())),
                                 u.set().b()),
                v.x()))

    def SolvePath(self, source, target, rounding=False, verbose=False, edges=None, preprocessing=False):
        assert len(source) == self.dimension
        assert len(target) == self.dimension

        # Add source and target vertices
        vertices = self.spp.Vertices()
        start = self.spp.AddVertex(Point(source), "start")
        goal = self.spp.AddVertex(Point(target), "goal")

        # Add edges connecting source and target to graph
        if edges is None:
            edges = self.findStartGoalEdges(source, target)
        source_connected = (len(edges[0]) > 0)
        target_connected = (len(edges[1]) > 0)

        for ii in edges[0]:
            u = vertices[ii]
            edge = self.spp.AddEdge(start, u, f"(start, {u.name()})")
            self.edge_cost_dict[edge.id()] = []

            for jj in range(self.dimension):
                edge.AddConstraint(start.x()[jj] == u.x()[jj])

        for ii in edges[1]:
            u = vertices[ii]
            edge = self.spp.AddEdge(u, goal, f"({u.name()}, goal)")

            edge_length = edge.AddCost(Binding[Cost](
                self.edge_cost, np.append(u.x(), goal.x())))[1]
            self.edge_cost_dict[edge.id()] = [edge_length]

        if not source_connected:
            raise ValueError('Source vertex is not connected.')
        if not target_connected:
            raise ValueError('Target vertex is not connected.')

        active_edges, result, hard_result, statistics = self.solveSPP(
            start, goal, rounding, preprocessing, verbose)

        if active_edges is None:
            self.ResetGraph([start, goal])
            return None, result, None, hard_result, statistics

        ########################################################################
        # The following overwrites the rounding times.

        from pydrake.all import MathematicalProgram, MosekSolver

        A = np.hstack((np.eye(self.dimension), -np.eye(self.dimension)))
        b = np.zeros(self.dimension)
        rounding_time = 0
        for i, path_edges in enumerate(active_edges):
            prog = MathematicalProgram()
            x = prog.NewContinuousVariables(len(path_edges), self.dimension)
            X = [e.u().set() for e in path_edges] + [path_edges[-1].v().set()]
            for Xv, xv in zip(X[:-1], x):
                Xv.AddPointInSetConstraints(prog, xv)
            for Xv, xv in zip(X[1:], x):
                Xv.AddPointInSetConstraints(prog, xv)
            for xu, xv in zip(x[:-1], x[1:]):
                s = prog.NewContinuousVariables(1)[0]
                prog.AddLinearCost(s)
                d = xv - xu
                prog.AddLorentzConeConstraint(s, d.dot(d))
            solver = MosekSolver()
            result = solver.Solve(prog)
            assert np.isclose(result.get_optimal_cost(), hard_result[i].get_optimal_cost(), rtol=1e-3, atol=1e-3)
            rounding_time += result.get_solver_details().optimizer_time
        statistics['rounding_time'] = rounding_time

        ########################################################################

        best_cost = np.inf
        best_path = None
        best_result = None
        for path, r in zip(active_edges, hard_result):
            if path is None or not r.is_success():
                continue
            if r.get_optimal_cost() < best_cost:
                best_cost = r.get_optimal_cost()
                best_path = path
                best_result = r

        if verbose:
            for edge in best_path:
                print("Added", edge.name(), "to path. Value:",
                      result.GetSolution(edge.phi()))

        # Extract trajectory
        waypoints = np.empty((self.dimension, 0))
        for edge in best_path:
            new_waypoint = best_result.GetSolution(edge.xv())
            waypoints = np.concatenate(
                [waypoints, np.expand_dims(new_waypoint, 1)], axis=1)

        self.ResetGraph([start, goal])
        return waypoints, result, best_result, hard_result, statistics
