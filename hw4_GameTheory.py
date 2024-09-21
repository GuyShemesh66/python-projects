# Skeleton file for HW4 question 4
# =====================================
# IMPORTANT: You are NOT allowed to modify the method signatures 
# (i.e. the arguments and return types each function takes). 
# We will pass your grade through an autograder which expects a specific format.
# =====================================


# Do not include any other files or an external package, unless it is one of
# [numpy, pandas, scipy, matplotlib, random]
# please contact us before sumission if you want another package approved.
import numpy as np
import matplotlib.pyplot as plt

# Implement the methods in this class as appropriate. Feel free to add other methods
# and attributes as needed. You may/should reuse code from previous HWs when applicable.
class DirectedGraph:
    def __init__(self, number_of_nodes):
        '''Assume that nodes are represented by indices/integers between 0 and number_of_nodes - 1.'''
        self.num_nodes = number_of_nodes
        self.adj_list = {i: [] for i in range(number_of_nodes)}
    
    def add_edge(self, origin_node, destination_node):
        '''Adds an edge from origin_node to destination_node.'''
        if destination_node not in self.adj_list[origin_node]:
            self.adj_list[origin_node].append(destination_node)
    
    def edges_from(self, origin_node):
        ''' This method shold return a list of all the nodes destination_node such that there is
            a directed edge (origin_node, destination_node) in the graph.'''
        return self.adj_list[origin_node]
    
    def get_edge(self, origin_node, destination_node):
        ''' This method should return true is there is an edge from origin_node to destination_node
        and false otherwise'''
        return destination_node in self.adj_list[origin_node]
    
    def number_of_nodes(self):
        ''' This method should return the number of nodes in the graph'''
        return self.num_nodes

# === Problem 6. ===
def scaled_page_rank(G, num_iter, eps = 1/7.0):
    ''' This method, given a DirectedGraph G, runs the epsilon-scaled 
    page-rank algorithm for num-iter iterations, for parameter eps,
    and returns a Dictionary where the keys are the set of 
    nodes [0,...,G.number_of_nodes() - 1], each associated with a value
    equal to the score of output by the eps-scaled pagerank algorithm.

    In the case of num_iter=0, all nodes should 
    have weight 1/G.number_of_nodes()'''
    n = G.number_of_nodes()
    scores = {i: 1/n for i in range(n)}
    
    for _ in range(num_iter):
        new_scores = {i: eps/n for i in range(n)}
        for v in range(n):
            out_edges = G.edges_from(v)
            if out_edges:
                for u in out_edges:
                    new_scores[u] += (1-eps) * scores[v] / len(out_edges)
            else:
                for u in range(n):
                    new_scores[u] += (1-eps) * scores[v] / n
        scores = new_scores
    
    return scores

def graph_15_1_left():
    ''' This method, should construct and return a DirectedGraph encoding the left example in fig 15.1
    Use the following indexes: A:0, B:1, C:2, Z:3 '''    
    G = DirectedGraph(4)
    G.add_edge(0, 1)  # A -> B
    G.add_edge(1, 2)  # B -> C
    G.add_edge(2, 0)  # C -> A
    G.add_edge(0, 3)  # A -> Z
    G.add_edge(3, 3)  # Z -> Z (self-loop)
    return G

def graph_15_1_right():
    ''' This method, should construct and return a DirectedGraph encoding the right example in fig 15.1
    Use the following indexes: A:0, B:1, C:2, Z1:3, Z2:4'''    
    G = DirectedGraph(5)
    G.add_edge(0, 1)  # A -> B
    G.add_edge(1, 2)  # B -> C
    G.add_edge(2, 0)  # C -> A
    G.add_edge(0, 3)  # A -> Z1
    G.add_edge(0, 4)  # A -> Z2
    G.add_edge(3, 4)  # Z1 -> Z2
    G.add_edge(4, 3)  # Z2 -> Z1
    return G

def graph_15_2():
    ''' This method, should construct and return a DirectedGraph encoding example 15.2
        Use the following indexes: A:0, B:1, C:2, A':3, B':4, C':5'''
    G = DirectedGraph(6)
    G.add_edge(0, 1)  # A -> B
    G.add_edge(1, 2)  # B -> C
    G.add_edge(2, 0)  # C -> A
    G.add_edge(3, 4)  # A' -> B'
    G.add_edge(4, 5)  # B' -> C'
    G.add_edge(5, 3)  # C' -> A'
    return G

def extra_graph_1():
    ''' This method, should construct and return a DirectedGraph of your choice with at least 10 nodes'''    
    G = DirectedGraph(10)
    for i in range(9):
        G.add_edge(i, i+1)
    G.add_edge(9, 0)
    return G

def extra_graph_2():
    ''' This method, should construct and return a DirectedGraph of your choice with at least 10 nodes'''    
    G = DirectedGraph(10)
    for i in range(10):
        for j in range(10):
            if i != j:
                G.add_edge(i, j)
    return G

# === Problem 8. ===
def facebook_graph(filename = "facebook_combined.txt"):
    ''' This method should return a DIRECTED version of the facebook graph as an instance of the DirectedGraph class.
    In particular, if u and v are friends, there should be an edge between u and v and an edge between v and u.'''    
    with open(filename, 'r') as f:
        edges = [tuple(map(int, line.strip().split())) for line in f]
    
    max_node = max(max(edge) for edge in edges)
    G = DirectedGraph(max_node + 1)
    
    for u, v in edges:
        G.add_edge(u, v)
        G.add_edge(v, u)
    
    return G


def _print_facebook_page_ranks(G, num_iter, eps):
    page_ranks = scaled_page_rank(G, num_iter, eps)
    top_10 = sorted(page_ranks.items(), key=lambda x: x[1], reverse=True)[:10]
    bottom_10 = sorted(page_ranks.items(), key=lambda x: x[1])[:10]
    print(f"<<<<<<<<<<<<<<< {num_iter} iterations and eps={eps} >>>>>>>>>>>>>>>>>")
    print(f"Top 10 nodes with scores:")
    for node, score in top_10:
        print(f"Node {node}: {score}")
    print(f"Bottom 10 nodes with scores:")
    for node, score in bottom_10:
        print(f"Node {node}: {score}")
    print("<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>")


def main():
    # Run the algorithms on all the graphs above
    print(scaled_page_rank(graph_15_1_left(), 10))
    print("graph_15_1_left with  eps= 1/100000 , iteration=100")
    print(scaled_page_rank(graph_15_1_left(), 100,1/100000.0))
    print(scaled_page_rank(graph_15_1_right(), 10))
    print(scaled_page_rank(graph_15_2(), 10))
    print(scaled_page_rank(extra_graph_1(), 10))
    print(scaled_page_rank(extra_graph_2(), 10))
    print("----------------------------------------------------------------------")
    G = facebook_graph()
    _print_facebook_page_ranks(G, 20, 1/7.0)
    _print_facebook_page_ranks(G, 50, 1/7.0)
    _print_facebook_page_ranks(G, 100, 1/7.0)
    # print(page_ranks)
    
    # Call the analysis function
    # analyze_facebook_pagerank()

if __name__ == "__main__":
    main()