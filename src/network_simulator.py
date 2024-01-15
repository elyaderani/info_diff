import logging
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd

from pathlib import Path
from datetime import datetime
from typing import Any, List, Iterable, Optional
from typing import Optional

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
Path(f"./data/sim_{timestamp}").mkdir(parents=True, exist_ok=True)

# parameters
simulation_params = {
    'N': 300,
    'T': 500,
    'gamma': 2,
    'delta': 0.5,
    'k': 4,  # number of immediate neighbors
    'p': 0.32,  # re-wiring probability
    'p_erdos_renyi': 0.032,  # probability of edge creation
    'epsilon': 1e-3,  # threshold for early stopping
    'convergence_period': 10,  # number of rounds to wait for convergence
    'seed_val': 1,  # seed for reproducibility 
    'timestamp': timestamp, # not used in the mcmc simulation
}

# configure logging
Path(f"./logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=f"./logs/sim_{timestamp}.log",
    format='%(asctime)s - %(message)s',
    level=logging.INFO)

# add logging to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


def create_network(network_type: str, n: int, k: int, p: float, p_erdos_renyi: float, seed: Optional[int]) -> nx.Graph:
    """Create a network of type 'network_type' with 'n' nodes and 'k' neighbors."""
    if seed:
        np.random.seed(seed)
    if network_type == 'newman_watts_strogatz':
        G = nx.newman_watts_strogatz_graph(n, k, p, seed=seed)
    elif network_type == 'barabasi_albert':
        G = nx.barabasi_albert_graph(n, k)
    elif network_type == 'erdos_renyi':
        G = nx.erdos_renyi_graph(n, p_erdos_renyi)
    elif network_type == 'watts_strogatz':
        G = nx.watts_strogatz_graph(n, k, p)
    else:
        raise ValueError(f'Network model {network_type} not supported!')

    t = np.random.uniform(size=n)
    nx.set_node_attributes(G, {i: {'type': t[i]} for i in range(n)})
    return G


def get_weight(delta: float, gamma: float, shortest_path_len: int, type_difference: float) -> float:
    """
    Compute the weight between two nodes based on their type and shortest path length.

    Parameters
    ----------
    delta: float
        The weight given to the shortest path length. Assumed to be in [0, 1].
    gamma: float
        The weight given to the type difference. Assumed to be >= 1.
    shortest_path_len: int
        The shortest path length between two nodes.
    type_difference: float
        The difference between the types of two nodes. Assumed to be in [0, 1].
    """
    return (delta*np.exp(-shortest_path_len)+(1-delta)*(1-type_difference))**gamma


def update_beliefs(W: np.ndarray, G: nx.Graph, beliefs: List[float], time: int, data: pd.DataFrame) -> List[float]:
    """
    Update the beliefs of all nodes in the network.

    Parameters
    ----------
    W: np.ndarray
        The weight matrix.
    G: nx.Graph
        The networkx graph.
    beliefs: List[float]
        The current beliefs of all nodes.
    time: int
        The current time step.
    data: pd.DataFrame
        The data frame to store the mcmc data.
    """
    new_beliefs = beliefs.copy()
    for i in range(G.number_of_nodes()):
        w_i, w_sum = [], []
        for j in range(G.number_of_nodes()): 
            if i != j:  # self-loop is disallowed
                if W[i, j] > np.random.uniform():
                    w_i.append((1 - abs(G.nodes[i]['type'] - G.nodes[j]['type'])) * beliefs[j])
                    w_sum.append(1 - abs(G.nodes[i]['type'] - G.nodes[j]['type']))
        if w_i:
            m_j = sum(w_i) / sum(w_sum)
            # rel = np.mean(w_sum)
            # new_beliefs[i] = (beliefs[i]**(1-rel) * m_j**rel) / ((beliefs[i]**(1-rel) * m_j**rel) + ((1 - beliefs[i])**(1-rel) * (1 - m_j)**rel))
            new_beliefs[i] = (beliefs[i] * m_j) / ((beliefs[i] * m_j) + ((1 - beliefs[i]) * (1 - m_j)))
        else:
            m_j = 0
            new_beliefs[i] = beliefs[i]
        
        data.append([time, i, m_j, new_beliefs[i], G.nodes[i]['type']])
        
    return new_beliefs, data


def draw_network(G: nx.Graph, beliefs: List[float], seed_val: int, title: str, directory: Path) -> None:
    """Draw the network with the nodes colored by their beliefs."""
    plt.figure(figsize=(12, 12))
    np.random.seed(seed_val)

    try:
        pos = nx.spring_layout(G, seed=seed_val)  # can change to other layouts
        nodes = nx.draw_networkx_nodes(G, pos, node_color=beliefs, cmap=plt.get_cmap('viridis'), alpha=0.6, vmin=0, vmax=1)
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        plt.colorbar(nodes)
        # plt.title(title)
        plt.axis('off')  # turn off axis
        plt.savefig(directory/f'{title}.png')
        plt.close()

    except Exception as e:
        logging.error(f'Error while drawing network: {e}')


def draw_evolution(steps: Iterable, beliefs_over_time: np.array, title: str, directory: Path) -> None:
    """Draw the evolution of beliefs over time."""
    plt.figure(figsize=(10, 5))
    plt.plot(steps, beliefs_over_time, '-', alpha=0.3)
    plt.plot(steps, np.mean(beliefs_over_time, axis=1), '--', label='mean', color='black')
    # plt.axvline(x=max(steps)-10, color='grey', linestyle='--', label='convergence')
    plt.xlabel('Time')
    plt.ylabel('Beliefs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory/f'{title}.png')
    plt.close()


def run_simulation(network_type: str, N: int, T: int, gamma:float, delta: float, k: int, p: float, p_erdos_renyi: float, epsilon: float, convergence_period: int, seed_val: int, timestamp: str) -> List[List[float]]:
    """
    Run the simulation for the given network type.

    Parameters
    ----------
    network_type: str
        The type of network to simulate.
    N: int
        The number of nodes in the network.
    T: int
        The number of time steps to simulate.
    gamma: float
        The weight given to the type difference. Assumed to be >= 1.
    delta: float
        The weight given to the shortest path length. Assumed to be in [0, 1].
    k: int
        The number of neighbors.
    p: float
        The re-wiring probability.
    p_erdos_renyi: float
        The probability of edge creation in the Erdos-Renyi model.
    epsilon: float
        The threshold for early stopping.
    convergence_period: int
        The number of rounds to wait for convergence.
    seed_val: int
        The seed for reproducibility.
    timestamp: str
        The timestamp of the simulation.
    """
    np.random.seed(seed_val)
    # create graph
    G = create_network(network_type, N, k, p, p_erdos_renyi,seed_val)
    for i in range(G.number_of_nodes()):
        G.nodes[i]['type'] = np.random.beta(.5, .5) #np.random.uniform() #
    # initialize beliefs, TODO: check if this is neccesary
    np.random.seed(seed_val)
    initial_beliefs = np.random.uniform(size=N)
    logging.info('')
    logging.info('-'*50)
    logging.info(f'Running simulation for {network_type} model...')
    logging.info(f'5 first initial beliefs: {initial_beliefs[:5]}')
    logging.info(
        f'N={N}, T={T}, delta={delta}, gamma={gamma}, k={k}, p={p}, convergence epsilon={epsilon}, seed={seed_val}')
    logging.info(f'mean clustering coefficient: {nx.average_clustering(G)}')
    logging.info(f'mean shortest path length: {nx.average_shortest_path_length(G)}')
    logging.info(f'mean degree: {(np.mean([G.degree[i] for i in range(N)]))}')
    logging.info(f'mean belief at time 0: {np.mean(initial_beliefs)}')
    logging.info(f'min belief at time 0: {np.min(initial_beliefs)}')
    logging.info(f'max belief at time 0: {np.max(initial_beliefs)}')
    logging.info(f'standard deviation of beliefs at time 0: {np.std(initial_beliefs)}')

    beliefs = initial_beliefs.copy()
    beliefs_over_time = [beliefs]
    D = np.array(nx.floyd_warshall_numpy(G))  # compute the full (shortest) path distance matrix
    W = np.zeros((N, N))  # define a weight matrix holder   
    # compute and replace the weights  
    for i in range(N):
        for j in range(N):
            if i != j:  # self-loop is disallowed
                shortest_path_len = D[i, j]
                # if no path exists, W[i, j] is 0
                if shortest_path_len == np.inf:
                    logging.info(f'No path exists between nodes {i} and {j}')
                    continue
                type_difference = abs(G.nodes[i]['type'] - G.nodes[j]['type'])
                W[i, j] = get_weight(delta, gamma, shortest_path_len, type_difference)     

    # propagate beliefs
    stagnant_counter = 0
    mcmc_data = []
    for t in range(1, T+1):
        old_beliefs = beliefs.copy()
        beliefs, mcmc_data = update_beliefs(W, G, beliefs, t, data = mcmc_data)
        beliefs_over_time.append(beliefs)
        # compute the change, in the l2 norm, for all beliefs
        dist = np.linalg.norm(old_beliefs - beliefs)
        if dist < epsilon: 
            stagnant_counter += 1
        else:
            stagnant_counter = 0
        # stop early if beliefs have been stagnant for 'convergence_period' rounds
        if stagnant_counter >= convergence_period:
            logging.info(f'Early stopping, beliefs reached steady state at time {t}...')
            break

    directory = Path(f"./data/sim_{timestamp}/{network_type}")
    directory.mkdir(parents=True, exist_ok=True)
    draw_network(G, beliefs_over_time[0],  seed_val, 'initial_state', directory)
    draw_network(G, beliefs_over_time[-1], seed_val, 'final_state', directory)
    draw_evolution(range(t+1), np.array(beliefs_over_time), 'belief_evolution', directory)
    beliefs_df = pd.DataFrame(np.array(beliefs_over_time).T)
    beliefs_df.to_csv(directory/'beliefs.csv', header=False, index=False)
    logging.info(f'Simulation ended for {network_type} model...')
    logging.info(f'Final mean belief: {np.mean(beliefs_over_time[-1])}')
    logging.info(f'Final min belief: {np.min(beliefs_over_time[-1])}')
    logging.info(f'Final max belief: {np.max(beliefs_over_time[-1])}')
    logging.info(f'Final standard deviation of beliefs: {np.std(beliefs_over_time[-1])}')
    logging.info('-'*50)

    return pd.DataFrame(mcmc_data, columns=['time', 'node', 'network_message', 'belief', 'type'])
    

if __name__ == '__main__':
    df = run_simulation('newman_watts_strogatz', **simulation_params)
    # pd.DataFrame(df).to_csv(f'./data/sim_{timestamp}/mcmc_newman_watts_strogatz.csv', index=False)
    # breakpoint()
    run_simulation('barabasi_albert', **simulation_params)
    # breakpoint()
    run_simulation('erdos_renyi', **simulation_params)
    # breakpoint()
    run_simulation('watts_strogatz', **simulation_params)