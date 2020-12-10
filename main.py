#!/usr/bin/env python3
"""
Implementation of a 1D finite element simulation to solve a transient heat
flow through a bar. Look in the `config.yaml` file for all the simulation's
controlling values. This project solves the weak form twice: with a backward
and forward Euler, and the errors are compared.
"""

import math
import pathlib
import re
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import yaml

_FIGURE_FOLDER = pathlib.Path.cwd() / "figures"


class BasisFunction:
    def __init__(self, lnode: float, rnode: float) -> None:
        self.lnode = lnode
        self.rnode = rnode
        self.node = (rnode + lnode) / 2

    def eval_local(self, x: float, verbose: bool = False) -> float:
        """Evaluate this function within a X \eplison [-1, 1] domain. This requires
        mapping the input from x -> X.

        Args:
            x: the global point to evaluate.

        Returns:
            The value of the derivative at this point.
        """

        X = (2 * x - (self.lnode + self.rnode)) / (self.rnode - self.lnode)

        if self.lnode <= x < self.node:
            return 1 + X
        elif self.node <= x <= self.rnode:
            return 1 - X
        else:
            return 0.0

    def eval_local_derivative(self, x: float) -> float:
        """Evaluate the derivate of this function within a X \eplison [-1, 1] domain.
        This requires mapping the input from x -> X.

        Args:
            x: the point to evaluate the local derivative in global coordinates.

        Returns:
            The value of the derivative at this point.
        """
        # Transform the global coordinate to local coordinates.
        # x(X) = 0.5*(lnode + rnode) + 0.5*(rnode - lnode)X
        # In english: start at the midpoint of the node then go
        # X*(half the width of the element) where X \epsilon [-1, 1]
        X = (2 * x - (self.lnode + self.rnode)) / (self.rnode - self.lnode)

        if self.lnode <= x < self.node:
            return 1.0 / (self.node - self.lnode)
        elif self.node <= x < self.rnode:
            return -1.0 / (self.rnode - self.node)
        else:
            return 0.0

    def local_to_global_J(self) -> float:
        """Return the Jacobian change of variable 'matrix'. Really just dx/dX."""
        return 0.5 * (self.rnode - self.lnode)


def parse_config(config: dict):
    """Parse the config given in the `config.yaml` file. Some default values
    will be used if the .yaml is missing important info.

    Args:
        config: The config from the `config.yaml` file.

    Returns:
        A dictionary of parsed information for the simulation.
    """
    config_parsed = {
        "euler_method": config.get("euler-method", "forward"),
        "dt": eval(config.get("time-step", "1/551")),
        "num_elements": config.get("num-elements", "11"),
        "bar_length": config.get("bar-length", 1),
    }
    assert config_parsed["euler_method"] in [
        "forward",
        "backward",
    ], "Specify 'backward' or 'forward' Euler!"
    config_parsed["dx"] = config_parsed["bar_length"] / (config_parsed["num_elements"])
    config_parsed["nodes"] = np.linspace(
        0, config_parsed["bar_length"], config_parsed["num_elements"] + 1
    )
    config_parsed["nodes_for_basis"] = np.linspace(
        -config_parsed["dx"],
        config_parsed["bar_length"] + config_parsed["dx"],
        2 + config_parsed["num_elements"] + 1,
    )
    config_parsed["nodes"] = np.round(config_parsed["nodes"], 2)
    config_parsed["nodes_for_basis"] = np.round(config_parsed["nodes_for_basis"], 2)
    config_parsed["elements"] = [
        (round(n1, 3), round(n2, 3))
        for n1, n2 in zip(config_parsed["nodes"][0::1], config_parsed["nodes"][1::1])
    ]

    boundary_conds = config.get("boundary-conditions", {})
    assert boundary_conds, "Please supply boundary conditions!"

    initial = boundary_conds.get("initial", {})
    assert initial, "Please supply initial conditions!"

    condition_matcher = re.compile(r"\((.*?)\)")
    # Split between equal sign
    left, right = initial.split("=")

    x, t = condition_matcher.search(left).group(1).split(",")
    config_parsed["initial_x"] = x
    config_parsed["initial_t"] = t
    config_parsed["initial_u"] = right.strip()
    config_parsed["forcing_function"] = config.get("forcing-function")

    dirichlet_left = boundary_conds.get("Dirichlet-left", "")
    dirichlet_right = boundary_conds.get("Dirichlet-right", "")
    assert (
        dirichlet_left and dirichlet_right
    ), "Please supply Dirichlet boundary conditions!"

    left, right = dirichlet_left.split("=")
    config_parsed["dirichlet_left"] = float(right)

    left, right = dirichlet_right.split("=")
    config_parsed["dirichlet_right"] = float(right)

    return config_parsed


def analytical_solution(x, t):
    return np.exp(-t) * np.sin(np.pi * x)


def solve_initial_condition(
    config: dict, basis_functions: List[BasisFunction], M: np.ndarray
):
    x = config["nodes"]
    u = np.array([eval(config["initial_u"])]).transpose()
    u[0] = config["dirichlet_left"]
    u[-1] = config["dirichlet_right"]
    c = np.matmul(np.linalg.inv(M), u)

    return u, c


def gaussian_quadrature(func, t, domain) -> float:
    """Perform Gaussian Quadrature integration on the _local_ element.
    In order to properly perform a local integration, we require the Jacobian
    change of variable matrix to be supplied."""
    # These are the local [-1, 1] guassian quadrature points for an element.
    # We transform them to the element's global domain and let the basis function
    # determine the values.
    Xs = [
        -(1 / 3) * np.sqrt(5 + (2 * np.sqrt(10 / 7))),
        -(1 / 3) * np.sqrt(5 - (2 * np.sqrt(10 / 7))),
        0,
        (1 / 3) * np.sqrt(5 - (2 * np.sqrt(10 / 7))),
        (1 / 3) * np.sqrt(5 + (2 * np.sqrt(10 / 7))),
    ]
    ws = [
        (322 - (13 * np.sqrt(70))) / 900,
        (322 + (13 * np.sqrt(70))) / 900,
        128 / 225,
        (322 + (13 * np.sqrt(70))) / 900,
        (322 - (13 * np.sqrt(70))) / 900,
    ]
    # Use element's global boundaries to transform these local quadrature points to
    # the global quadrature. This is a little bit redundant, but we do it so the basis
    # function can determine whether or not these values are in its domain.
    xs = [0.5 * (domain[0] + domain[1]) + 0.5 * (domain[1] - domain[0]) * x for x in Xs]

    return sum(wi * func(xi, t) for xi, wi in zip(xs, ws))


def create_M(
    basis_functions: List[BasisFunction], elements: List[Tuple[float, float]]
) -> np.ndarray:
    """Create the mass matrix."""

    M = np.zeros((len(basis_functions), len(basis_functions)))

    # What we do here is loop over the basis functions and peform the integration of the
    # over the reference element. We use a _mapping_ here to evaluate each integral
    # locally.
    for i, func_i in enumerate(basis_functions):
        for j, func_j in enumerate(basis_functions):
            f = (
                lambda x, t: func_i.eval_local(x)
                * func_j.eval_local(x)
                * func_i.local_to_global_J()
            )

            # Integrate over the entire bar in an element by element fashion
            values = [gaussian_quadrature(f, None, element) for element in elements]
            M[i][j] = sum(values)

    return M


def create_K(
    basis_functions: List[BasisFunction], elements: List[Tuple[float, float]]
) -> np.ndarray:
    """Create the stiffness matrix."""
    K = np.zeros((len(basis_functions), len(basis_functions)))

    # What we do here is loop over the basis functions and perform the integration of the
    # over the reference element. We use a _mapping_ here to evaluate each integral
    # locally.
    for i, func_i in enumerate(basis_functions):
        for j, func_j in enumerate(basis_functions):
            f = (
                lambda x, t: func_i.eval_local_derivative(x)
                * func_j.eval_local_derivative(x)
                * func_i.local_to_global_J()
            )
            # Loop over each element
            values = [gaussian_quadrature(f, None, element) for element in elements]
            K[i][j] = sum(values)

    return K


def plot_surface(config: dict, analytical_solution, u_sim: np.ndarray, fig_name: str) -> None:
    """Function to plot the surface of the analytical solution and the simulation
    solution.

    Args:
        config: The governing config for the program.
        analytical_solution: The system's analytical solution.
        u_sim: the temperature values produced by the simulation.
    """
    fig = plt.figure(figsize=plt.figaspect(0.5))

    x = config["nodes"]
    t = np.arange(0, 1 + config["dt"], config["dt"])
    x, t = np.meshgrid(x, t)

    u_a = analytical_solution(x, t)
    X = config["nodes"]
    T = np.arange(0, 1 + config["dt"], config["dt"])
    X, T = np.meshgrid(X, T)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("t", fontsize=10)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("u", fontsize=10)
    ax.set_title("Simulated Temperature")
    surf = ax.plot_surface(
        X, T, u_sim, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    surf = ax.plot_surface(x, t, u_a, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("t", fontsize=10)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("u", fontsize=10)
    ax.set_title("Analytical Temperature")
    fig.savefig(_FIGURE_FOLDER / fig_name)


def solve(config: dict) -> None:
    """Main entrypoint function that will solve create all the necessary components
    to solve the problem.

    Args:
        config: The dictionary obtained from `config.yaml`.
    """

    config = parse_config(config)
    x_global = config["nodes"]
    basis_functions = []
    for idx, node in enumerate(config["nodes_for_basis"][1:-1]):
        func = BasisFunction(
            config["nodes_for_basis"][idx], config["nodes_for_basis"][idx + 2]
        )
        basis_functions.append(func)

    # Plot the basis functions for inspection
    for idx, basis_func in enumerate(basis_functions):
        xs = [
            config["nodes_for_basis"][idx],
            (config["nodes_for_basis"][idx] + config["nodes_for_basis"][idx + 2]) / 2,
            config["nodes_for_basis"][idx + 2],
        ]
        ys = [
            basis_func.eval_local(xs[0]),
            basis_func.eval_local(xs[1]),
            basis_func.eval_local(xs[2]),
        ]
        plt.plot(xs, ys, color="blue")
    plt.scatter(config["nodes"], np.zeros_like(config["nodes"]), c="red")

    x = config["nodes"]

    # Create the mass matrix
    M = create_M(basis_functions, config["elements"])
    M[0, :] = 0.0
    M[-1, :] = 0.0
    M[0, 0] = 1.0
    M[-1, -1] = 1.0
    # Create the stiffness matrix
    K = create_K(basis_functions, config["elements"])
    print(f"M matrix:\n{M}")
    print(f"K matrix:\n{K}")

    average_losses = None
    # Perform the **Forward Euler**
    if config["euler_method"] == "forward":
        print("Solving Forward Euler.")
        u0, c = solve_initial_condition(config, basis_functions, M)
        c = u0

        u_forward = np.zeros((math.ceil(1 / config["dt"] + 1), len(config["nodes"])))
        u_forward[0] = u0[:, 0]
        forward_losses = []
        for idx_t, t in enumerate(
            np.arange(config["dt"], 1 + config["dt"], config["dt"]), start=1
        ):
            # NOTE: this will evaluate the string in the config. x and t need to be
            # defined before this call.
            forcing = []
            for basis in basis_functions:
                # Construct the function to be integrated.
                f = (
                    lambda x, t: basis.eval_local(x)
                    * (eval(config["forcing_function"]))
                    * basis.local_to_global_J()
                )
                values = [
                    gaussian_quadrature(f, t, element) for element in config["elements"]
                ]
                forcing.append(sum(values))

            forcing = np.array([forcing]).transpose()

            b = (
                np.matmul(M, c)
                - config["dt"] * np.matmul(K, c)
                + config["dt"] * forcing
            )
            b[0] = config["dirichlet_left"]
            b[-1] = config["dirichlet_right"]
            M_imposed = M
            M_imposed[0, :] = 0.0
            M_imposed[-1, :] = 0.0
            M_imposed[0, 0] = 1.0
            M_imposed[-1, -1] = 1.0

            c = np.matmul(np.linalg.inv(M_imposed), b)
            real_sol = np.expand_dims(analytical_solution(x, t), axis=-1)
            forward_losses.append(np.power(c - real_sol, 2))
            for idx, node in enumerate(config["nodes"]):
                u_forward[idx_t, idx] = sum(
                    ci * basis_func.eval_local(node)
                    for ci, basis_func in zip(c, basis_functions)
                )
        forward_losses = np.array(forward_losses)

        average_losses = np.array([np.mean(loss) for loss in forward_losses])

        """
        # This plots the end temperature for forward euler
        fig = plt.figure()
        plt.plot(x, forward_losses[-1], label="Forward Euler Temperature")
        plt.legend()
        plt.xlabel("Node Number (x direction)")
        plt.ylabel("Temperature")
        plt.title("Temperature at t=1.0 in Forward Euler")
        """
        figure_scatter = plt.figure()
        plt.scatter(
            np.linspace(config["dt"], 1.0, math.ceil(1 / config["dt"])),
            average_losses,
            label="Forward Euler",
        )
        plt.legend(fontsize=10)
        plt.xlabel("time", fontsize=15)
        plt.ylabel("Average L2 Error Over Bar", fontsize=15)
        plt.title(
            f"Average L2 Error vs time (\u0394t\u2248{config['dt']:.5f})", fontsize=15
        )
        figure_scatter.savefig(
            _FIGURE_FOLDER / f"{config['euler_method']}_average_l2_error.png"
        )
        plot_surface(config, analytical_solution, u_forward, "forward_euler_surface.png")

    if config["euler_method"] == "backward":
        u0, c = solve_initial_condition(config, basis_functions, M)
        c = u0
        u_backward = np.zeros((math.ceil(1 / config["dt"] + 1), len(config["nodes"])))
        u_backward[0] = u0[:, 0]
        # Perform the **Backward Euler**
        print("Solving Backward Euler.")
        backward_losses = []
        for idx_t, t in enumerate(
            np.arange(config["dt"], 1 + config["dt"], config["dt"]), start=1
        ):
            # NOTE: this will evaluate the string in the config. x and t need to be
            # defined before this call.
            forcing = []
            for basis in basis_functions:
                values = []
                f = lambda x, t: (
                    basis.eval_local(x, True)
                    * (eval(config["forcing_function"]))
                    * basis.local_to_global_J()
                )

                values = [
                    gaussian_quadrature(f, t, element) for element in config["elements"]
                ]
                forcing.append(sum(values))

            forcing = np.array([forcing]).transpose()

            b = np.matmul(M, c) + config["dt"] * forcing
            b[0] = 0
            b[-1] = 0
            M_imposed = M
            M_imposed[0, :] = 0.0
            M_imposed[-1, :] = 0.0
            M_imposed[0, 0] = 1.0
            M_imposed[-1, -1] = 1.0
            c = np.matmul(np.linalg.inv(M_imposed + config["dt"] * K), b)
            real_sol = np.expand_dims(analytical_solution(x, t), axis=-1)
            backward_losses.append(np.power(c - real_sol, 2))
            for idx, node in enumerate(config["nodes"]):
                values = [
                    ci * basis_func.eval_local(node)
                    for ci, basis_func in zip(c, basis_functions)
                ]
                u_backward[idx_t, idx] = sum(values)
        backward_losses = np.array(backward_losses)

        average_losses = np.array([np.mean(loss) for loss in backward_losses])

        figure_scatter = plt.figure()
        plt.scatter(
            np.linspace(config["dt"], 1.0, math.ceil(1 / config["dt"])),
            average_losses,
            label="Backward Euler",
        )
        plt.legend(fontsize=10)
        plt.xlabel("time", fontsize=15)
        plt.ylabel("Average L2 Error Over Bar", fontsize=15)
        plt.title(
            f"Average L2 Error vs time (\u0394t\u2248{config['dt']:.5f})", fontsize=15
        )
        figure_scatter.savefig(
            _FIGURE_FOLDER / f"{config['euler_method']}_average_l2_error.png"
        )
        plot_surface(config, analytical_solution, u_backward, "backward_euler_surface.png")


if __name__ == "__main__":
    # Read in the config.
    config_file = pathlib.Path(__file__).parent / "config.yaml"
    assert (
        config_file.is_file()
    ), "Can't find the config. Place the config.yaml in the same folder as 'main.py'"
    config = yaml.safe_load(config_file.read_text())

    _FIGURE_FOLDER.mkdir(exist_ok=True)

    solve(config)
