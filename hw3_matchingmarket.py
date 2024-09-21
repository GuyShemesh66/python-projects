# Skeleton file for HW3 questions 7 and 8
# =====================================
# IMPORTANT: You are NOT allowed to modify the method signatures 
# (i.e. the arguments and return types each function takes). 
# We will pass your grade through an autograder which expects a specific format.
# =====================================
# Do not include any other files or an external package, unless it is one of
# [numpy, pandas, scipy, matplotlib, random]
# please contact us before submission if you want another package approved.
import numpy as np
import matplotlib.pyplot as plt
import random


####### All the code in this section was taken from our previous assignment ########
class WeightedDirectedGraph:
    def __init__(self, number_of_nodes):
        '''Assume that nodes are represented by indices/integers between 0 and number_of_nodes - 1.'''
        self._number_of_nodes = number_of_nodes
        self._adj_list = [[] for _ in range(number_of_nodes)]
    
    def set_edge(self, origin_node, destination_node, weight=1):
        ''' Modifies the weight for the specified directed edge, from origin to destination node,
            with specified weight (an integer >= 0). If weight = 0, effectively removes the edge from 
            the graph. If edge previously wasn't in the graph, adds a new edge with specified weight.'''
        # Remove existing edge if present
        self._adj_list[origin_node] = [edge for edge in self._adj_list[origin_node] if edge[0] != destination_node]
        if weight > 0:
            self._adj_list[origin_node].append((destination_node, weight))
    
    def edges_from(self, origin_node):
        ''' This method should return a list of all the nodes destination_node such that there is
            a directed edge (origin_node, destination_node) in the graph (i.e. with weight > 0).'''
        return [edge[0] for edge in self._adj_list[origin_node]]
    
    def get_edge(self, origin_node, destination_node):
        ''' This method should return the weight (an integer > 0) 
            if there is an edge between origin_node and 
            destination_node, and 0 otherwise.'''
        for node, weight in self._adj_list[origin_node]:
            if node == destination_node:
                return weight
        return 0
    
    def number_of_nodes(self):
        return self._number_of_nodes

def _bfs(graph, s, t):
    visited = [False] * graph.number_of_nodes()
    parent = [-1] * graph.number_of_nodes()
    queue = [s]
    visited[s] = True
    
    while queue:
        u = queue.pop(0)
        for v in graph.edges_from(u):
            if not visited[v] and graph.get_edge(u, v) > 0:
                queue.append(v)
                visited[v] = True
                parent[v] = u
                if v == t:
                    return parent
    return parent

def _find_path_flow(graph, s, t, parent):
    path_flow = float('inf')
    v = t
    while v != s:
        u = parent[v]
        path_flow = min(path_flow, graph.get_edge(u, v))
        v = u
    return path_flow

def _update_residual_graph(graph, s, t, parent, path_flow):
    v = t
    while v != s:
        u = parent[v]
        graph.set_edge(u, v, graph.get_edge(u, v) - path_flow)
        graph.set_edge(v, u, graph.get_edge(v, u) + path_flow)
        v = u

def _max_flow(G, s, t):
    '''Given a WeightedDirectedGraph G, a source node s, a destination node t,
       compute the (integer) maximum flow from s to t, treating the weights of G as capacities.
       Return a tuple (v, F) where v is the integer value of the flow, and F is a maximum flow
       for G, represented by another WeightedDirectedGraph where edge weights represent
       the final allocated flow along that edge.'''
    # Create a residual graph and fill it with capacities from the original graph
    residual_graph = WeightedDirectedGraph(G.number_of_nodes())
    for u in range(G.number_of_nodes()):
        for v in G.edges_from(u):
            residual_graph.set_edge(u, v, G.get_edge(u, v))

    max_flow_value = 0
    parent = [-1] * G.number_of_nodes()

    while True:
        parent = _bfs(residual_graph, s, t)
        if parent[t] == -1:  # No augmenting path found
            break
        
        path_flow = _find_path_flow(residual_graph, s, t, parent)
        if path_flow == 0:  # No more flow can be pushed
            break
        
        max_flow_value += path_flow
        _update_residual_graph(residual_graph, s, t, parent, path_flow)

    # Construct the flow graph based on the residual graph
    F = WeightedDirectedGraph(G.number_of_nodes())
    for u in range(G.number_of_nodes()):
        for v in G.edges_from(u):
            flow = G.get_edge(u, v) - residual_graph.get_edge(u, v)
            if flow > 0:
                F.set_edge(u, v, flow)

    return max_flow_value, F



######################## Below starts our real assignments ##########################################################
def _build_graph_from_matrix(n, m, C, s, t):
    """
    Build a WeightedDirectedGraph from the given constraint matrix C.
    """
    G = WeightedDirectedGraph(t + 1)  # t is the highest node index
    
    # Add edges from source to left vertices
    for i in range(n):
        G.set_edge(s, i, 1)
    
    # Add edges from right vertices to sink
    for j in range(m):
        G.set_edge(n+ j, t, 1)
    
    # Add edges between left and right vertices based on C
    for i in range(n):
        for j in range(m):
            if C[i][j] == 1:
                G.set_edge(i, n+j, 1)
    
    return G


def _generate_all_permutations(elements, r):
    if r == 0:
        yield []
    elif r > len(elements):
        return
    else:
        for i in range(len(elements)):
            for permutation in _generate_all_permutations(elements[i+1:], r-1):
                yield [elements[i]] + permutation


def _find_constricted_sets(n, C):
    numbers_range = list(range(n))
    for current_number in range(n + 1):
        for subset in _generate_all_permutations(numbers_range, current_number):
            subset_neighbors = _find_neighborhood(subset, C)
            if len(subset) > len(subset_neighbors):
                return subset
    return []

# === Problem 7(a) ===
def _matching_or_cset_for_non_square_matrix(n, m, C):
    """
    This method is very similar to the method defined above but with support for non
    square matrices. We were not allowed to changed the signature for the method above, so we've added
    this method with the modified signature
    """
    s, t = n + m, n + m + 1
    G = _build_graph_from_matrix(n, m, C, s, t)
    flow, F = _max_flow(G, s, t)
    
    if flow == min(n, m):
        # Perfect matching exists
        M = [None] * n
        for i in range(n):
            for j in range(m):
                if F.get_edge(i, n + j) == 1:
                    M[i] = j
                    break
        return (True, M)
    else:
        # Find constricted set using BFS
        return False, _find_constricted_sets(n, C)

# === Problem 7(a) ===
def matching_or_cset(n, C):
    '''Given a bipartite graph, with 2n vertices, with
    n vertices in the left part, and n vertices in the right part,
    and edge constraints C, output a tuple (b, M) where b is True iff 
    there is a matching, and M is either a matching or a constricted set.
    -   Specifically, C is a n x n array, where
        C[i][j] = 1 if left vertex i (in 0...n-1) and right vertex j (in 0...n-1) 
        are connected by an edge. If there is no edge between vertices
        (i,j), then C[i][j] = 0.
    -   If there is a perfect matching, return an n-element list M where M[i] = j 
        if driver i is matched with rider j.
    -   If there is no perfect matching, return a list M of left vertices
        that comprise a constricted set.
    '''
    return _matching_or_cset_for_non_square_matrix(n, n, C)


# === Problem 7(b) ===
def _compute_preferred_choice_graph(n, m, V, P):
    """Compute the preferred choice graph based on valuations and prices."""
    C = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        max_utility = max(V[i][k] - P[k] for k in range(m))
        for j in range(m):
            if V[i][j] - P[j] == max_utility:
                C[i][j] = 1
    return C

def _find_neighborhood(S, C):
    """Find the neighborhood N(S) of set S in graph C."""
    N_S = set()
    for i in S:
        for j in range(len(C[i])):
            if C[i][j] != 0:
                N_S.add(j)
    return N_S

def _update_prices(P, N_S):
    """Increase prices for items in N(S) and shift all prices."""
    for j in N_S:
        P[j] += 1
    min_price = min(P)
    return [p - min_price for p in P]

def market_eq(n, m, V):
    '''Given a matching market with n buyers and m items, and 
    valuations V, output a market equilibrium tuple (P, M)
    of prices P and a matching M.
    -   Specifically, V is an n x m list, where
        V[i][j] is a number representing buyer i's value for item j.
    -   Return a tuple (P, M), where P is an m-element list, where 
        P[j] equals the price of item j.
        M is an n-element list, where M[i] = j if buyer i is 
        matched with item j, and M[i] = None if there is no matching.
    In sum, buyer i receives item M[i] and pays P[M[i]].'''
    original_n, original_m = n, m
    if n < m:
        # Add fake buyers as shown in the recitation
        V += [[0] * m for _ in range(m - n)]
        n = m
    elif m < n:
        # Add fake items as shown in the recitation
        for row in V:
            row += [0] * (n - m)
        m = n
    P = [0] * m
    M = [None] * n
    
    while True:
        C = _compute_preferred_choice_graph(n, m, V, P)
        is_perfect, result = matching_or_cset(n, C)
        if is_perfect:
            M = result
            # Convert matches with fake items/buyers to None
            M = [j if j < original_m else None for j in M[:original_n]]
            P = P[:original_m]  # Only keep original items' prices
            return (P, M)
        else:
            S = result
            N_S = _find_neighborhood(S, C)
            P = _update_prices(P, N_S)

# === Problem 8(b) ===
def _social_value(M, V):
    return sum(V[i][M[i]] for i in range(len(M)) if M[i] is not None)

def _delete_row(matrix, row):
    return matrix[:row] + matrix[row+1:]

def vcg(n, m, V):
    '''Given a matching market with n buyers, and m items, and
    valuations V as defined in market_eq, output a tuple (P,M)
    of prices P and a matching M as computed using the VCG mechanism
    with Clarke pivot rule.
    V,P,M are defined equivalently as for market_eq. Note that
    P[j] should be positive for every j in 0...m-1. Note that P is
    still indexed by item, not by player!!
    '''
    # Find optimal matching
    _, M = market_eq(n, m, V)
    optimal_value = _social_value(M, V)

    # Compute VCG prices
    VCG_P = [0] * m
    for i in range(n):
        if M[i] is not None:
            # Compute optimal matching without buyer i
            V_without_i = _delete_row(V, i)
            _, M_without_i = market_eq(n-1, m, V_without_i)
            value_without_i = _social_value(M_without_i, V_without_i)
            
            # Compute VCG price
            others_welfare_without_i = value_without_i
            others_welfare_with_i = optimal_value - V[i][M[i]]
            externality = others_welfare_without_i - others_welfare_with_i
            VCG_P[M[i]] = max(0, externality)

    return (VCG_P, M)

# === Bonus Question 2(a) (Optional) ===
def random_bundles_valuations(n, m):
    '''Given n buyers, m bundles, generate a matching market context
    (n, m, V) where V[i][j] is buyer i's valuation for bundle j.
    Each bundle j (in 0...m-1) is comprised of j copies of an identical good.
    Each player i has its own value for an individual good; this value is sampled
    uniformly at random from [1, 50] inclusive, for each player'''
    V = [[0] * m for _ in range(n)]
    for i in range(n):
        value_per_good = random.randint(1, 50)
        for j in range(m):
            V[i][j] = value_per_good * (j + 1)
    return (n, m, V)

# === Bonus Question 2(b) (optional) ===
def gsp(n, m, V):
    '''Given a matching market for bundles with n buyers, and m bundles, and
    valuations V (for bundles), output a tuple (P, M) of prices P and a 
    matching M as computed using GSP.'''
    sorted_buyers = sorted(range(n), key=lambda i: V[i][m-1], reverse=True)
    
    M = [None] * n
    P = [0] * m
    
    for i, buyer in enumerate(sorted_buyers):
        if i < m:
            M[buyer] = i
            if i < m - 1 and i + 1 < n:
                P[i] = V[sorted_buyers[i+1]][i] // (i + 1)
    
    return (P, M)

# This is needed for question 2c
def best_response(player, bids, valuations, num_items):
    """
    Calculate the best response bids for a player given the current bids and valuations.
    """
    other_player = 1 - player
    best_bids = bids[player][:]
    for item in range(num_items):
        if bids[player][item] > valuations[player][item]:
            if bids[other_player][item] < valuations[player][item]:
                # Keep the current bid if the competitor's bid is lower than the valuation
                best_bids[item] = bids[player][item]
            elif bids[other_player][item] > valuations[player][item]:
                # Lower the bid below the competitor's bid
                best_bids[item] = max(1, bids[other_player][item] - 1)
            else:
                # Keep the current bid
                best_bids[item] = bids[player][item]
        elif valuations[player][item] > bids[other_player][item]:
            if bids[player][item] > bids[other_player][item]:
                # Keep the bid if the player's bid is already higher
                best_bids[item] = bids[player][item]
            else:
                # Bid slightly above the other player's bid
                best_bids[item] = bids[other_player][item] + 1
        else:
            # Do not change the bid if the other player's bid is higher than the player's valuation
            best_bids[item] = bids[player][item]
    
    return best_bids

def _gsp_brd(num_players, num_items, valuations, bids, iterations=1000):
    """
    Run the BRD process until convergence or max iterations.
    """
    print(f"Number of Players: {num_players}")
    print(f"Number of Items: {num_items}")
    print(f"Initial valuations: {valuations}")
    print(f"Initial bids: {bids}")
    i = 0
    for iteration in range(iterations):
        i += 1
        new_bids = [bid[:] for bid in bids]
        for player in range(len(bids)):
            best_bids = best_response(player, bids, valuations, num_items)
            if best_bids != bids[player]:
                print(f"Player {player + 1} changes their bids from {bids[player]} to {best_bids}")
            new_bids[player] = best_bids
        if new_bids == bids:
            return new_bids, True, i-1
        bids = new_bids
    return bids, False, i

def main():
    print("\nTest case from Lecture 5 Page 7")
    n, m = 3, 3
    V = [[4, 12, 5], [7, 10, 9], [7, 7, 10]]
    market_eq_prices, market_eq_matching = market_eq(n, m, V)  # test 7b
    vcg_prices, vcg_matching = vcg(n, m, V)  # test 8a
    print("Question 7(b):")
    print("Market equilibrium prices:", market_eq_prices)
    print("Market equilibrium matching:", market_eq_matching)
    print("\nTest case from Lecture 5 Page 7")
    print("Question 8(a):")
    print("VCG prices:", vcg_prices)
    print("VCG matching:", vcg_matching)
    # Question 8(b): Compare VCG and market_eq prices
    n, m = 5, 15
    n_values = [5, 10, 20]
    m_values = [5, 10, 20]
    
    print("\nQuestion 8(b): Comparing VCG and market_eq prices")
    for n in n_values:
        for m in m_values:
            print("n = ", n, "m = ", m)
            V = [[random.randint(1, 100) for _ in range(m)] for _ in range(n)]
            market_eq_prices, _ = market_eq(n, m, V)
            vcg_prices, _ = vcg(n, m, V)
            print("VCG prices:", vcg_prices)
            print("Market equilibrium prices:", market_eq_prices)

    # Bonus Question 2(a): Random bundles valuations
    print("\nBonus Question 2(a): Random bundles valuations")
    n_bundles, m_bundles = 20, 20
    bundle_n, bundle_m, bundle_V = random_bundles_valuations(n_bundles, m_bundles)
    vcg_bundle_prices, vcg_bundle_matching = vcg(bundle_n, bundle_m, bundle_V)
    print("VCG prices for bundles:", vcg_bundle_prices)

    # Bonus Question 2(b): GSP pricing
    print("\nBonus Question 2(b): GSP pricing")
    gsp_prices, gsp_matching = gsp(bundle_n, bundle_m, bundle_V)
    print("GSP prices:", gsp_prices)
    print("VCG prices:", vcg_bundle_prices)

    # Bonus Question 2(c): BRD on GSP
    print("\nBonus Question 2(c): BRD on GSP")
    num_players = 2
    num_items = 2
    valuations = [[random.randint(10, 100) for _ in range(num_items)] for _ in range(num_players)]
    bids = [[random.randint(1, max(player_vals)) for _ in range(num_items)] for player_vals in valuations]
    final_bids, converged, it = _gsp_brd(num_players,num_items,valuations,bids)
    print(f"Converged: {converged}")
    print(f"Final bids: {final_bids}")
    print(f"Number of iterations: {it}")


if __name__ == "__main__":
    main()