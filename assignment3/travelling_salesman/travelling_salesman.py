import random
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely.geometry import LineString


class Node:
    def __init__(self, id, x, y):
        self.id = id - 1
        self.x = x
        self.y = y

    @staticmethod
    def get_distance_between_nodes(node1, node2):
        return np.linalg.norm(np.subtract(node1.pos, node2.pos))

    @property
    def pos(self):
        return self.x, self.y


def from_listofstrings(data):
    return Node(*tuple(map(lambda n: int(float(n)), data)))


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

    def switch(self):
        valid = False

        while not valid:
            (a, b), (c, d) = random.sample(self.graph.edges, 2)
            self._switch_edges(a, b, c, d)
            valid = self._validate()
            if not valid:
                self._switch_edges(a, b, c, d, revert=True)

        return a, b, c, d

    def move(self):
        valid = False

        while not valid:
            a = random.choice(self.nodes).id
            b = a
            c = a
            while b == a or c == a:
                b, c = random.choice(list(self.graph.edges))

            a_n1, a_n2 = self.graph.neighbors(a)
            
            self._move_node(a, b, c, a_n1, a_n2)
            valid = self._validate()
            if not valid:
                self._move_node(a, b, c, a_n1, a_n2, revert=True)

        return a, b, c, a_n1, a_n2

    def two_opt(self, n=1000):
        for _ in range(n):
            changed = False

            every_edge_combination = list(combinations(self.graph.edges, 2))
            random.shuffle(every_edge_combination)

            for (a, b), (c, d) in every_edge_combination:
                node_a, node_b, node_c, node_d = self.nodes[a], self.nodes[b], self.nodes[c], self.nodes[d]

                line_ab = LineString([node_a.pos, node_b.pos])
                line_cd = LineString([node_c.pos, node_d.pos])
                line_ac = LineString([node_a.pos, node_c.pos])
                line_bd = LineString([node_b.pos, node_d.pos])

                if line_ab.intersects(line_cd):
                    if line_ab.length + line_cd.length > line_ac.length + line_bd.length:
                        self._switch_edges(a, b, c, d)
                        valid = self._validate()
                        if valid:
                            changed = True
                            break
                        else:
                            self._switch_edges(a, b, c, d, revert=True)

            if not changed:
                # print('no intersecting edges can be changed')
                break

    def _validate(self):
        return nx.is_connected(self.graph)

    def _switch_edges(self, a, b, c, d, revert=False):
        b, c = (c, b) if revert else (b, c)
        if not (self.graph.has_edge(a, c) or self.graph.has_edge(b, d)):
            self.graph.remove_edge(a, b)
            self.graph.remove_edge(c, d)
            self.graph.add_edge(a, c, distance=Node.get_distance_between_nodes(self.nodes[a], self.nodes[c]))
            self.graph.add_edge(b, d, distance=Node.get_distance_between_nodes(self.nodes[b], self.nodes[d]))

    def _move_node(self, a, b, c, a_n1, a_n2, revert=False):
        if revert:
            tmp_b, tmp_c = b, c
            b, c = a_n1, a_n2
            a_n1, a_n2 = tmp_b, tmp_c

        self.graph.remove_edge(a, a_n1)
        self.graph.remove_edge(a, a_n2)
        self.graph.remove_edge(b, c)
        self.graph.add_edge(a_n1, a_n2, distance=Node.get_distance_between_nodes(self.nodes[a_n1], self.nodes[a_n2]))
        self.graph.add_edge(a, b, distance=Node.get_distance_between_nodes(self.nodes[a], self.nodes[b]))
        self.graph.add_edge(a, c, distance=Node.get_distance_between_nodes(self.nodes[a], self.nodes[c]))

    def get_total_distance(self):
        return sum([edge[2]['distance'] for edge in self.graph.edges(data=True)])

    def draw_graph(self, draw_edges=False):
        fig, ax = plt.subplots()
        nx.draw_networkx_nodes(self.graph, pos=self.positions, ax=ax)
        weights = [self.graph[u][v]['distance'] for u, v in self.graph.edges()]
        nx.draw_networkx_edges(self.graph, pos=self.positions, width=np.array(weights) / 50) if draw_edges else None
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.show()
