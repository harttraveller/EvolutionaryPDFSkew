import numpy as np
from scipy.stats import skewnorm
from scipy import integrate
from tqdm import tqdm
import matplotlib.pyplot as plt


def gen_pdf(params):
    return lambda x: skewnorm.pdf(
        x, a=params["a"], loc=params["loc"], scale=params["scale"]
    )


def compute_dist(params, X):
    return gen_pdf(params)(X)


def compute_loss(params, inputs, X, edge_target, edge_exp, modal_exp):
    constant_mult = 1000
    left = abs(integrate.quad(gen_pdf(params), -float("inf"), inputs["lower"])[0])
    left = abs(edge_target - left) * constant_mult
    left = left**edge_exp
    right = abs(integrate.quad(gen_pdf(params), inputs["upper"], float("inf"))[0])
    right = abs(edge_target - right) * constant_mult
    right = right**edge_exp
    dist = compute_dist(params, X)
    modal = abs(X[np.argmax(dist)] - inputs["mode"]) * constant_mult
    modal = modal**modal_exp
    return (right + left + modal) / 3


def mutate_params(params, a_mut, loc_mut, scale_mut):
    params = params.copy()
    params["a"] = params["a"] * abs(np.random.normal(1, a_mut))
    params["loc"] = params["loc"] * abs(np.random.normal(1, loc_mut))
    params["scale"] = params["scale"] * abs(np.random.normal(1, scale_mut))
    return params


def generate_mutations(params, a_mut, loc_mut, scale_mut, n_mutations):
    mutated_params = [
        mutate_params(params, a_mut, loc_mut, scale_mut) for _ in range(n_mutations)
    ]
    return mutated_params


def gen_domain(lower, upper, params, resolution):
    X = np.linspace(lower - params["scale"], upper + params["scale"], resolution)
    return X


def evolve_params(
    lower,
    mode,
    upper,
    resolution,
    a_mut,
    loc_mut,
    scale_mut,
    n_mutations,
    edge_target,
    iters,
    edge_exp,
    modal_exp,
):
    logs = {"params": list(), "dists": list(), "loss": list()}
    inputs = {"lower": lower, "mode": mode, "upper": upper}
    params = {"a": 1, "loc": mode, "scale": ((mode - lower) + (upper - mode)) / 2}
    logs["params"].append(params)
    X = gen_domain(lower, upper, params, resolution)
    for _ in tqdm(range(iters)):
        mutations = generate_mutations(params, a_mut, loc_mut, scale_mut, n_mutations)
        param_losses = [
            compute_loss(params, inputs, X, edge_target, edge_exp, modal_exp)
            for params in mutations
        ]
        params = mutations[param_losses.index(min(param_losses))]
        logs["params"].append(params)
        logs["dists"].append(compute_dist(params, X))
        logs["loss"].append(min(param_losses))
    return params, logs
