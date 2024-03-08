import numpy as np
from tqdm import tqdm, trange

# def likelihood(x, beta):
#     # Standard Normal Distribution
#     # An underlying assumption of linear regression is that the residuals
#     # are Gaussian Normal Distributed; often, Standard Normal distributed
#     return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)


# def proposal_distribution(x, stepsize=0.5):
#     # Select the proposed state (new guess) from a Gaussian distriution
#     # centered at the current state, within a Guassian of width `stepsize`
#     return np.random.normal(x, stepsize)


def mcmc_updater(curr_state, curr_likeli, likelihood, proposal_distribution):
    """ Propose a new state and compare the likelihoods
    
    Args:
        curr_state (float): the current parameter/state value
        curr_likeli (float): the current likelihood estimate
        likelihood (function): a function handle to compute the likelihood
        proposal_distribution (function): a function handle to compute the 
          next proposal state

    Returns:
        (tuple): either the current state or the new state
          and its corresponding likelihood
    """
    # Generate a proposal state using the proposal distribution
    # Proposal state == new guess state to be compared to current
    # print(curr_state)
    proposal_state = proposal_distribution(curr_state)

    # Calculate the acceptance criterion
    prop_likeli = likelihood(proposal_state)
    accept_crit = prop_likeli / curr_likeli

    # print("prop_likeli = ", prop_likeli ,"curr_likeli = ", curr_likeli)
    if np.abs(curr_likeli) < 1e-15:
      accept_crit = 1
    # Generate a random number between 0 and 1
    accept_threshold = np.random.uniform(0, 1)

    # If the acceptance criterion is greater than the random number,
    # accept the proposal state as the current state
    if accept_crit > accept_threshold:
        # print("accepted.")
        return proposal_state, prop_likeli

    # Else
    return curr_state, curr_likeli

def metropolis_hastings(
        likelihood, proposal_distribution, initial_state, 
        num_samples, stepsize=0.5, burnin=0.2):
    """ Compute the Markov Chain Monte Carlo

    Args:
        likelihood (function): a function handle to compute the likelihood
        proposal_distribution (function): a function handle to compute the 
          next proposal state
        initial_state (list): The initial conditions to start the chain
        num_samples (integer): The number of samples to compte, 
          or length of the chain
        burnin (float): a float value from 0 to 1.
          The percentage of chain considered to be the burnin length

    Returns:
        samples (list): The Markov Chain,
          samples from the posterior distribution
    """
    samples = []

    # The number of samples in the burn in phase
    idx_burnin = int(burnin * num_samples)

    # Set the current state to the initial state
    curr_state = initial_state
    curr_likeli = likelihood(curr_state)

    for i in tqdm(range(num_samples)):
        # The proposal distribution sampling and comparison
        #   occur within the mcmc_updater routine
        curr_state, curr_likeli = mcmc_updater(
            curr_state=curr_state,
            curr_likeli=curr_likeli,
            likelihood=likelihood,
            proposal_distribution=proposal_distribution
        )

        # Append the current state to the list of samples
        if i >= idx_burnin:
            # Only append after the burnin to avoid including
            #   parts of the chain that are prior-dominated
            samples.append(curr_state)

    return samples


# def likelihood(x):
#     # Standard Normal Distribution
#     # An underlying assumption of linear regression is that the residuals
#     # are Gaussian Normal Distributed; often, Standard Normal distributed
#     return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)


# def proposal_distribution(x, stepsize=0.5):
#     # Select the proposed state (new guess) from a Gaussian distriution
#     # centered at the current state, within a Guassian of width `stepsize`
#     return np.random.normal(x, stepsize)
