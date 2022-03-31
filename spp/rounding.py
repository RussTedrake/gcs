import networkx as nx
import numpy as np

def depthFirst(source, target, candidateEdges, selector):
    visited_vertices = [source]
    path_vertices = [source]
    path_edges = []
    while path_vertices[-1] != target:
        candidate_edges = candidateEdges(path_vertices[-1], visited_vertices)
        if len(candidate_edges) == 0:
            path_vertices.pop()
            path_edges.pop()
        else:
            next_edge, next_vertex = selector(candidate_edges)
            visited_vertices.append(next_vertex)
            path_vertices.append(next_vertex)
            path_edges.append(next_edge)
    return path_edges

def incomingEdges(gcs):
    incoming_edges = {v.id(): [] for v in gcs.Vertices()}
    for e in gcs.Edges():
        incoming_edges[e.v().id()].append(e)
    return incoming_edges

def outgoingEdges(gcs):
    outgoing_edges = {u.id(): [] for u in gcs.Vertices()}
    for e in gcs.Edges():
        outgoing_edges[e.u().id()].append(e)
    return outgoing_edges

def optimalFlows(gcs, result):
    return {e.id(): result.GetSolution(e.phi()) for e in gcs.Edges()}

def greedyEdgeSelector(candidate_edges, flows):
    candidate_flows = [flows[e.id()] for e in candidate_edges]
    return candidate_edges[np.argmax(candidate_flows)]

def randomEdgeSelector(candidate_edges, flows):
    candidate_flows = np.array([flows[e.id()] for e in candidate_edges])
    probabilities = candidate_flows/sum(candidate_flows)
    return np.random.choice(candidate_edges, p=probabilities)

def greedyForwardPathSearch(gcs, result, source, target, flow_tol=1e-5, **kwargs):
    print('greedyForwardPathSearch')

    outgoing_edges = outgoingEdges(gcs)
    flows = optimalFlows(gcs, result)

    def candidateEdges(current_vertex, visited_vertices):
        keepEdge = lambda e: e.v() not in visited_vertices and flows[e.id()] > flow_tol
        return [e for e in outgoing_edges[current_vertex.id()] if keepEdge(e)]

    def selector(candidate_edges):
        e = greedyEdgeSelector(candidate_edges, flows)
        return e, e.v()

    return [depthFirst(source, target, candidateEdges, selector)]

def randomForwardPathSearch(gcs, result, source, target, num_paths=10, seed=None, flow_tol=1e-5, **kwargs):
    print('randomForwardPathSearch')

    if seed is not None:
        np.random.seed(seed)

    outgoing_edges = outgoingEdges(gcs)
    flows = optimalFlows(gcs, result)

    def candidateEdges(current_vertex, visited_vertices):
        keepEdge = lambda e: e.v() not in visited_vertices and flows[e.id()] > flow_tol
        return [e for e in outgoing_edges[current_vertex.id()] if keepEdge(e)]

    def selector(candidate_edges):
        e = randomEdgeSelector(candidate_edges, flows)
        return e, e.v()

    return [depthFirst(source, target, candidateEdges, selector) for _ in range(num_paths)]

def greedyBackwardPathSearch(gcs, result, source, target, flow_tol=1e-5, **kwargs):
    print('greedyBackwardPathSearch')

    incoming_edges = incomingEdges(gcs)
    flows = optimalFlows(gcs, result)

    def candidateEdges(current_vertex, visited_vertices):
        keepEdge = lambda e: e.u() not in visited_vertices and flows[e.id()] > flow_tol
        return [e for e in incoming_edges[current_vertex.id()] if keepEdge(e)]

    def selector(candidate_edges):
        e = greedyEdgeSelector(candidate_edges, flows)
        return e, e.u()

    return [depthFirst(target, source, candidateEdges, selector)[::-1]]

def randomBackwardPathSearch(gcs, result, source, target, num_paths=10, seed=None, flow_tol=1e-5, **kwargs):
    print('randomBackwardPathSearch')

    if seed is not None:
        np.random.seed(seed)

    incoming_edges = incomingEdges(gcs)
    flows = optimalFlows(gcs, result)

    def candidateEdges(current_vertex, visited_vertices):
        keepEdge = lambda e: e.u() not in visited_vertices and flows[e.id()] > flow_tol
        return [e for e in incoming_edges[current_vertex.id()] if keepEdge(e)]

    def selector(candidate_edges):
        e = randomEdgeSelector(candidate_edges, flows)
        return e, e.u()

    return [depthFirst(target, source, candidateEdges, selector)[::-1] for _ in range(num_paths)]

def MipPathExtraction(gcs, result, source, target, **kwargs):
    return greedyForwardPathSearch(gcs, result, source, target)

def averageVertexPositionSpp(gcs, result, source, target, edge_cost_dict=None, flow_min=1e-3, **kwargs):
    print('averageVertexPositionSpp')

    G = nx.DiGraph()
    G.add_nodes_from(gcs.Vertices())

    vertex_data = {}
    for v in gcs.Vertices():
        vertex_data[v.id()] = np.zeros(v.set().ambient_dimension() + 1)

    for e in gcs.Edges():
        vertex_data[e.u().id()][:-1] += e.GetSolutionPhiXu(result)
        vertex_data[e.u().id()][-1] += result.GetSolution(e.phi())
        if e.v() == target:
            vertex_data[target.id()][:-1] += e.GetSolutionPhiXv(result)
            vertex_data[target.id()][-1] += result.GetSolution(e.phi())

    for v in gcs.Vertices():
        if vertex_data[v.id()][-1] > flow_min:
            vertex_data[v.id()] = vertex_data[v.id()][:-1] / vertex_data[v.id()][-1]
        else:
            vertex_data[v.id()] = v.set().ChebyshevCenter()

    for e in gcs.Edges():
        G.add_edge(e.u(), e.v())
        e_cost = 0
        for cost in edge_cost_dict[e.id()]:
            if len(cost.variables()) == e.u().set().ambient_dimension():
                e_cost += cost.evaluator().Eval(vertex_data[e.u().id()])
            elif len(cost.variables()) == e.u().set().ambient_dimension() + e.v().set().ambient_dimension():
                e_cost += cost.evaluator().Eval(np.append(vertex_data[e.u().id()], vertex_data[e.v().id()]))
            else:
                raise Exception("Unclear what variables are used in this cost.")
        G.edges[e.u(), e.v()]['l'] = np.squeeze(e_cost)
        if G.edges[e.u(), e.v()]['l'] < 0:
            raise RuntimeError(f"Averaged length of edge {e} is negative. Consider increasing flow_min.")

    path_vertices = nx.dijkstra_path(G, source, target, 'l')

    path_edges = []
    for u, v in zip(path_vertices[:-1], path_vertices[1:]):
        for e in gcs.Edges():
            if e.u() == u and e.v() == v:
                path_edges.append(e)
                break

    return [path_edges]
