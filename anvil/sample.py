"""
sample.py

Module containing functions related to sampling from a trained model
"""

from math import exp, isfinite, ceil

import numpy as np
import torch

from tqdm import tqdm

from reportengine import collect


def sample_batch(loaded_model, action, batch_size, current_state=None):
    r"""
    Sample using Metroplis-Hastings algorithm from a large number of phi
    configurations.

    We calculate the condition

        A = min[1, (\tilde p(phi^i) * p(phi^j)) / (p(phi^i) * \tilde p(phi^j))]

    Where i is the index of the current phi in metropolise chain and j is the
    current proposal. A uniform random number, u, is drawn and if u <= A then
    the proposed state phi^j is accepted (and becomes phi^i for the next update)

    Parameters
    ----------
    loaded_model: Module
        loaded_model which is going to be used to generate sample states
    action: Module
        the action upon which the loaded_model was trained, used to calculate the
        acceptance condition
    batch_size: int
        the number of states to generate from the loaded_model
    current_state: torch.Tensor or None
        the current state of the chain. None if this is the first batch

    Returns
    -------
    phi[chain_indices, :]: torch.Tensor
        chain of configurations generated by the MH algorithm
    history: torch.BoolTensor
        boolean tensor containing accept/reject history of chain
    """
    with torch.no_grad():  # don't track gradients
        z = torch.randn(
            (batch_size + 1, loaded_model.size_in)
        )  # random z configurations
        phi = loaded_model.inverse_map(z)  # map using trained loaded_model to phi
        if current_state is not None:
            phi[0] = current_state
        log_ptilde = loaded_model(phi)
    history = torch.zeros(batch_size, dtype=torch.bool)  # accept/reject history
    chain_indices = torch.zeros(batch_size, dtype=torch.long)

    log_ratio = log_ptilde + action(phi)
    if not isfinite(exp(float(min(log_ratio) - max(log_ratio)))):
        raise ValueError("could run into nans")

    i = 0  # phi index of current state
    for j in range(1, batch_size + 1):  # j = phi index of proposed state
        condition = min(1, exp(float(log_ratio[i] - log_ratio[j])))
        if np.random.uniform() <= condition:  # accepted
            chain_indices[j - 1] = j
            history[j - 1] = True
            i = j
        else:  # rejected
            chain_indices[j - 1] = i

    return phi[chain_indices, :], history


def thermalised_state(loaded_model, action) -> torch.Tensor:
    r"""
    A (hopefully) short initial sampling phase to allow the system to thermalise.

    Parameters
    ----------
    loaded_model: Module
        loaded_model which is going to be used to generate sample states
    action: Module
        the action upon which the loaded_model was trained, used to calculate the
        acceptance condition

    Returns
    -------
    states[-1]: torch.Tensor
        the final phi state
    """
    t_therm = 10000  # ideally come up with a way of working this out on the fly
    #t_therm = 1

    states, _ = sample_batch(loaded_model, action, t_therm)

    print(f"Thermalisation: discarded {t_therm} configurations.")
    return states[-1]


def chain_autocorrelation(loaded_model, action, thermalised_state) -> float:
    r"""
    Compute an observable-independent measure of the integrated autocorrelation
    time for the Markov chain.

        \tau_int = 0.5 + sum_{\tau=1}^{\tau_max} \rho(\tau)/\rho(0)

    where \rho(\tau)/\rho(0) is the probability of \tau consecutive rejections,
    which we estimate by

        \rho(\tau)/\rho(0) = # consecutive runs of \tau rejections / (N - \tau)

    See eqs. (16) and (19) in https://arxiv.org/pdf/1904.12072.pdf

    This measure of autocorrelation is used to provide a first guess for an
    appropriate subsampling interval,

        sample_interval = ceil(2 * integrated_autocorrelation)

    with the intended effect being that observables on the subsampled chain
    are entirely decorrelated.

    See http://luscher.web.cern.ch/luscher/lectures/LesHouches09.pdf section 2.2.4

    Parameters
    ----------
    loaded_model: Module
        loaded_model which is going to be used to generate sample states
    action: Module
        the action upon which the loaded_model was trained, used to calculate the
        acceptance condition
    initial_state:
        the current state of the Markov chain, after thermalisation

    Returns
    -------
    sample_interval: float
        Guess for subsampling interval, based on the integrated autocorrelation time

    """
    # Hard coded num states for estimating integrated autocorrelation
    batch_size = 10000

    # Sample some states
    states, history = sample_batch(loaded_model, action, batch_size, thermalised_state)

    N = len(history)
    autocorrelations = torch.zeros(N + 1, dtype=torch.float)  # +1 in case 100% rejected
    consecutive_rejections = 0

    for step in history:
        if step == True:  # move accepted
            if consecutive_rejections > 0:  # faster than unnecessarily accessing array
                autocorrelations[1 : consecutive_rejections + 1] += torch.arange(
                    consecutive_rejections, 0, -1, dtype=torch.float
                )
            consecutive_rejections = 0
        else:  # move rejected
            consecutive_rejections += 1
    if consecutive_rejections > 0:  # pick up last rejection run
        autocorrelations[1 : consecutive_rejections + 1] += torch.arange(
            consecutive_rejections, 0, -1, dtype=torch.float
        )

    # Compute integrated autocorrelation
    integrated_autocorrelation = 0.5 + torch.sum(
        autocorrelations / torch.arange(N + 1, 0, -1, dtype=torch.float)
    )
    print(f"Integrated autocorrelation time: {integrated_autocorrelation}")

    sample_interval = ceil(2 * integrated_autocorrelation)
    print(
        f"Guess for sampling interval: {sample_interval}, based on {batch_size} configurations."
    )

    return sample_interval


def sample(
    loaded_model, action, target_length: int, thermalised_state, chain_autocorrelation
) -> torch.Tensor:
    r"""
    Produces a Markov chain with approximately target_length decorrelated configurations,
    using the Metropolis-Hastings algorithm.

    Parameters
    ----------
    loaded_model: Module
        loaded_model which is going to be used to generate sample states
    action: Module
        the action upon which the loaded_model was trained, used to calculate the
        acceptance condition
    target_length: int
        the desired number of states to generate from the loaded_model

    Returns
    -------
    decorrelated_chain: torch.Tensor
        a sample of states from loaded_model, size = (target_length, loaded_model.size_in)

    """

    # Thermalise
    current_state = thermalised_state

    # Calculate sampling interval from integrated autocorrelation time
    sample_interval = chain_autocorrelation
    #sample_interval = 1

    # Decide how many configurations to generate, in order to get approximately
    # target_length after picking out decorrelated configurations
    batch_size = min(target_length, 10000)  # hard coded for now
    dec_samp_per_batch = ceil(batch_size / sample_interval)
    batch_size = dec_samp_per_batch * sample_interval
    Nbatches = ceil(target_length / dec_samp_per_batch)
    actual_length = dec_samp_per_batch * Nbatches

    decorrelated_chain = torch.empty(
        (actual_length, loaded_model.size_in), dtype=torch.float32
    )
    accepted = 0

    print(
        f"Generating {Nbatches * batch_size} configurations "
        f"in {Nbatches} batches of size {batch_size}"
    )

    pbar = tqdm(range(Nbatches), desc="batch")
    for batch in pbar:
        # Generate sub-chain of batch_size configurations
        batch_chain, batch_history = sample_batch(
            loaded_model, action, batch_size, current_state
        )
        current_state = batch_chain[-1]

        accepted += torch.sum(batch_history)

        # Add to larger chain
        start = batch * dec_samp_per_batch
        decorrelated_chain[start : start + dec_samp_per_batch, :] = batch_chain[
            ::sample_interval
        ]

    # Accept-reject statistics
    rejected = Nbatches * batch_size - accepted
    fraction = accepted / float(accepted + rejected)
    print(f"Accepted: {accepted}, Rejected: {rejected}, Fraction: " f"{fraction:.2g}")

    print(f"Returning a decorrelated chain of length: {actual_length}")

    return decorrelated_chain


sample_training_output = collect("sample", ("training_context",))
