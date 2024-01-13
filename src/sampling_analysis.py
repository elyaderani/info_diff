import arviz as az
import logging
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime


# simulation script
from network_simulator import run_simulation

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
log_dir = Path(".").resolve() / "data" / f"mcmc_{timestamp}"
log_dir.mkdir(parents=True, exist_ok=True)

# parameters
simulation_params = {
    'N': 300,
    'T': 500,
    'gamma': 2,
    'delta': 0.5,
    'k': 4,  # number of immediate neighbors
    'p': 0.3,  # re-wiring probability
    'epsilon': 1e-3,  # threshold for early stopping
    'convergence_period': 10,  # number of rounds to wait for convergence
    'seed_val': 1,  # seed for reproducibility 
    'timestamp': timestamp, # not used in the mcmc simulation
}

mcmc_params = {
    'sample_size': 4000, # posterior samples
    'tune_size': 2000,
    'n_chains': 2,
    'n_nodes': 4,
}


# configure logging
logging.basicConfig(
    filename=log_dir / "mcmc.log",
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)

# add logging to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)



def run_mcmc_study(simulation_params, mcmc_params):
    """
    Run the MCMC study for the given simulation parameters

    Parameters
    ----------
    simulation_params : dict
        Simulation parameters
    mcmc_params : dict
        MCMC parameters

    Returns
    -------
    mcmc_results : dict
        MCMC results for each node
    """
    try:
        simulation_results = run_simulation('newman_watts_strogatz', **simulation_params)
    except FileNotFoundError:
        logging.error("Could not read the data, check the path")
        return
    
    logging.info(f'Data shape: {simulation_results.shape}')
    logging.info(f'Data columns: {simulation_results.columns}')
    
    unique_nodes = simulation_results['node'].unique()
    selected_nodes = np.random.choice(unique_nodes, size=mcmc_params['n_nodes'], replace=False)
    logging.info(f'Selected nodes: {selected_nodes}')

    mcmc_results = {}

    for node in selected_nodes:
        filtered_data = simulation_results[simulation_results['node'] == node]
        logging.info(f'Node {node} size: {len(filtered_data)}')

        with pm.Model() as model:
            alpha = pm.Normal('alpha', mu=0, sd=1)
            beta_message = pm.Normal('beta_message', mu=0, sd=1)
            messages = filtered_data['network_message'].shift(1)[1:]
            curr_beliefs = filtered_data['belief'][1:]
            
            # prediction = alpha + beta_message * messages
            prediction = alpha + beta_message * filtered_data['belief'].shift(1)[1:]
            pm.Normal('belief', mu=prediction, sd=0.1, observed=curr_beliefs)

            trace = pm.sample(
                mcmc_params['sample_size'], tune=mcmc_params['tune_size'], 
                chains=mcmc_params['n_chains'], random_seed=simulation_params['seed_val']
                )

        sum_data = az.summary(trace, hdi_prob=0.94)
        logging.info(sum_data)

        az.plot_forest(trace)

        plt.savefig(log_dir / f'forest_node_{node}.png', bbox_inches='tight')
        az.plot_trace(trace, compact=True, combined=True, legend=True,figsize=(24, 16))
        plt.tight_layout()
        plt.savefig(log_dir / f'traceplot_node_{node}.png', bbox_inches='tight')

        plt.close() # close to avoid memory leaks

        mcmc_results[node] = {
            'trace': trace,
            'summary': sum_data,
        }

    return mcmc_results

if __name__ == "__main__":
    run_mcmc_study(simulation_params, mcmc_params)

