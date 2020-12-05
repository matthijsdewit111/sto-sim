from travelling_salesman.travelling_salesman import *

base_path = 'travelling_salesman/TSP-Configurations/'
data_files = ['eil51.tsp.txt', 'a280.tsp.txt', 'pcb442.tsp.txt']


def read_tsp_file(file: str = data_files[0]) -> [Node]:
    file_path = base_path + file
    with open(file_path, 'r') as f:
        name = f.readline().strip().split()[1]  # NAME
        comment = f.readline().strip().split()[1]  # COMMENT
        file_type = f.readline().strip().split()[1]  # TYPE
        dimension = f.readline().strip().split()[1]  # DIMENSION
        edge_weight_type = f.readline().strip().split()[1]  # EDGE_WEIGHT_TYPE
        f.readline()
        nodes = [Node.from_listofstrings(node) for node in [line.strip().split() for line in f.readlines()[:-1]]]
        f.close()
    return nodes


def simulated_annealing():
    nodes = read_tsp_file()
    ts = TravellingSalesman(nodes)
    print(f'\ntotal distance before two_opt: {ts.get_total_distance():3f}')
    ts.two_opt()
    print(f'\ntotal distance after two_opt: {ts.get_total_distance():3f}')
    ts.draw_graph(draw_edges=True)


if __name__ == '__main__':
    simulated_annealing()
