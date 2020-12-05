import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from shapely.geometry import LineString


class Node:
    def __init__(self, id, x, y):
        self.id = id - 1
        self.x = x
        self.y = y

    @staticmethod
    def get_distance_between_nodes(node1, node2):
        return np.linalg.norm(np.subtract(node1.pos, node2.pos))

    @classmethod
    def from_listofstrings(cls, data):
        return cls(*tuple(map(lambda n: int(float(n)), data)))

    @property
    def pos(self):
        return self.x, self.y


class TravellingSalesman:
    def __init__(self, nodes):
        self.nodes = nodes
        self.number_of_nodes = len(nodes)
        self.graph = self._construct_graph()

        self.positions = {}
        for node_id in self.graph.nodes:
            self.positions[node_id] = self.nodes[node_id].pos

    def _construct_graph(self):
        graph = nx.Graph()

        visited = []
        while len(visited) < self.number_of_nodes:
            rnode = random.choice(self.nodes)
            if rnode.id not in visited:
                visited.append(rnode.id)

        visited.append(visited[0])  # add connection between final node and starting node
        nx.add_path(graph, visited)

        edge_weights = {}
        for (i, j, _) in graph.edges(data=True):
            distance = Node.get_distance_between_nodes(self.nodes[i], self.nodes[j])
            edge_weights[i, j] = {'distance': distance}
        nx.set_edge_attributes(graph, edge_weights)

        for node in self.nodes:
            graph.nodes[node.id]['pos'] = node.pos

        return graph

    def two_opt(self, n=10000):
        for _ in range(n):
            edges = list(self.graph.edges)
            edge1 = random.choice(edges)
            edge2 = random.choice(edges)
            if edge1 == edge2:
                continue
            a, b, c, d = edge1[0], edge1[1], edge2[0], edge2[1]
            node_a, node_b, node_c, node_d = self.nodes[a], self.nodes[b], self.nodes[c], self.nodes[d]

            line_ab = LineString([node_a.pos, node_b.pos])
            line_cd = LineString([node_c.pos, node_d.pos])
            line_ac = LineString([node_a.pos, node_c.pos])
            line_bd = LineString([node_b.pos, node_d.pos])

            if line_ab.intersects(line_cd):
                if line_ab.length + line_cd.length > line_ac.length + line_bd.length:
                    self._switch_edges(a, b, c, d)
                    self._validate_graph(a, b, c, d)

    def _validate_graph(self, a, b, c, d):
        if nx.is_connected(self.graph):
            for i in self.graph.nodes():
                if len(self.graph[i]) != 2:
                    self._switch_edges(a, b, c, d, revert=True)  # undo switch
                    break
        else:
            self._switch_edges(a, b, c, d, revert=True)  # undo switch

    def _switch_edges(self, a, b, c, d, revert=False):
        b, c = (c, b) if revert else (b, c)
        if not (self.graph.has_edge(a, c) or self.graph.has_edge(b, d)):
            self.graph.remove_edge(a, b)
            self.graph.remove_edge(c, d)
            self.graph.add_edge(a, c, distance=Node.get_distance_between_nodes(self.nodes[a], self.nodes[c]))
            self.graph.add_edge(b, d, distance=Node.get_distance_between_nodes(self.nodes[b], self.nodes[d]))

    def get_total_distance(self):
        return sum([edge[2]['distance'] for edge in self.graph.edges(data=True)])

    def draw_graph(self, draw_edges=False):
        fig, ax = plt.subplots()
        nx.draw_networkx_nodes(self.graph, pos=self.positions, ax=ax)
        weights = [self.graph[u][v]['distance'] for u, v in self.graph.edges()]
        nx.draw_networkx_edges(self.graph, pos=self.positions, width=np.array(weights) / 50) if draw_edges else None
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.show()
