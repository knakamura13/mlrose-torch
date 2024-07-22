"""Functions to implement the randomized optimization and search algorithms."""

# Author: Genevieve Hayes
# License: BSD 3 clause

import numpy as np

from .opt_probs import DiscreteOpt, ContinuousOpt, TSPOpt
from .decay import GeomDecay, ArithDecay, ExpDecay, CustomSchedule


def hill_climb(
    problem: DiscreteOpt | ContinuousOpt | TSPOpt,
    max_iters: int | float = np.inf,
    restarts: int = 0,
    init_state: np.ndarray = None,
    curve: bool = False,
    random_state: int = None,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, np.ndarray]:
    """Use standard hill climbing to find the optimum for a given optimization problem.

    Parameters
    ----------
    problem: DiscreteOpt | ContinuousOpt | TSPOpt
        Object containing fitness function optimization problem to be solved.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm for each restart.
    restarts: int, default: 0
        Number of random restarts.
    init_state: array, default: None
        1-D Numpy array containing starting state for algorithm.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used by np.random.seed();
        otherwise, the random seed is not set.

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: array
        Numpy array containing the fitness at every iteration. Only returned if input argument `curve` is `True`.
    """
    if (not isinstance(max_iters, int) and max_iters != np.inf) or max_iters < 0:
        raise ValueError("max_iters must be a positive integer or np.inf.")
    if not isinstance(restarts, int) or restarts < 0:
        raise ValueError("restarts must be a positive integer.")
    if init_state is not None and len(init_state) != problem.get_length():
        raise ValueError("init_state must have same length as problem.")

    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    best_fitness = -np.inf
    best_state = None
    fitness_curve = []

    for _ in range(restarts + 1):
        if init_state is None:
            problem.reset()
        else:
            problem.set_state(init_state)

        iters = 0

        while iters < max_iters:
            iters += 1
            problem.find_neighbors()
            next_state = problem.best_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            current_fitness = problem.get_fitness()
            if next_fitness > current_fitness:
                problem.set_state(next_state)
            else:
                break

            if curve:
                fitness_curve.append(current_fitness)

        final_fitness = problem.get_fitness()
        if final_fitness > best_fitness:
            best_fitness = final_fitness
            best_state = problem.get_state()

    best_fitness *= problem.get_maximize()

    if curve:
        return best_state, best_fitness, np.asarray(fitness_curve)

    return best_state, best_fitness


def random_hill_climb(
    problem: DiscreteOpt | ContinuousOpt | TSPOpt,
    max_attempts: int = 10,
    max_iters: int | float = np.inf,
    restarts: int = 0,
    init_state: np.ndarray = None,
    curve: bool = False,
    random_state: int = None,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, np.ndarray]:
    """Use randomized hill climbing to find the optimum for a given optimization problem.

    Parameters
    ----------
    problem: DiscreteOpt | ContinuousOpt | TSPOpt
        Object containing fitness function optimization problem to be solved.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better neighbor at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    restarts: int, default: 0
        Number of random restarts.
    init_state: array, default: None
        1-D Numpy array containing starting state for algorithm.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used by np.random.seed();
        otherwise, the random seed is not set.

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: array
        Numpy array containing the fitness at every iteration. Only returned if input argument `curve` is `True`.
    """
    if not isinstance(max_attempts, int) or max_attempts < 0:
        raise ValueError("max_attempts must be a positive integer.")
    if (not isinstance(max_iters, int) and max_iters != np.inf) or max_iters < 0:
        raise ValueError("max_iters must be a positive integer or np.inf.")
    if not isinstance(restarts, int) or restarts < 0:
        raise ValueError("restarts must be a positive integer.")
    if init_state is not None and len(init_state) != problem.get_length():
        raise ValueError("init_state must have same length as problem.")

    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    best_fitness = -np.inf
    best_state = None
    fitness_curve = []

    for _ in range(restarts + 1):
        if init_state is None:
            problem.reset()
        else:
            problem.set_state(init_state)

        attempts = 0
        iters = 0

        while attempts < max_attempts and iters < max_iters:
            iters += 1
            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            current_fitness = problem.get_fitness()

            # If best neighbor is an improvement OR equal to current fitness (for plateau traversal),
            # move to that state and reset attempts counter
            if next_fitness >= current_fitness:
                problem.set_state(next_state)
                attempts = 0
            else:
                attempts += 1

            if curve:
                fitness_curve.append(current_fitness)

        final_fitness = problem.get_fitness()
        if final_fitness > best_fitness:
            best_fitness = final_fitness
            best_state = problem.get_state()

    best_fitness *= problem.get_maximize()

    if curve:
        return best_state, best_fitness, np.asarray(fitness_curve)

    return best_state, best_fitness


def simulated_annealing(
    problem: DiscreteOpt | ContinuousOpt | TSPOpt,
    schedule: GeomDecay | ArithDecay | ExpDecay | CustomSchedule = GeomDecay(),
    max_attempts: int = 10,
    max_iters: int | float = np.inf,
    init_state: np.ndarray = None,
    curve: bool = False,
    random_state: int = None,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, np.ndarray]:
    """Use simulated annealing to find the optimum for a given optimization problem.

    Parameters
    ----------
    problem: DiscreteOpt | ContinuousOpt | TSPOpt
        Object containing fitness function optimization problem to be solved.
    schedule: GeomDecay | ArithDecay | ExpDecay | CustomSchedule, default: GeomDecay()
        Schedule used to determine the value of the temperature parameter.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better neighbor at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    init_state: array, default: None
        1-D Numpy array containing starting state for algorithm.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used by np.random.seed();
        otherwise, the random seed is not set.

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: array
        Numpy array containing the fitness at every iteration. Only returned if input argument `curve` is `True`.
    """
    if not isinstance(max_attempts, int) or max_attempts < 0:
        raise ValueError("max_attempts must be a positive integer.")
    if (not isinstance(max_iters, int) and max_iters != np.inf) or max_iters < 0:
        raise ValueError("max_iters must be a positive integer or np.inf.")
    if init_state is not None and len(init_state) != problem.get_length():
        raise ValueError("init_state must have same length as problem.")

    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    if init_state is None:
        problem.reset()
    else:
        problem.set_state(init_state)

    fitness_curve = []
    attempts = 0
    iters = 0

    while attempts < max_attempts and iters < max_iters:
        temp = schedule.evaluate(iters)
        iters += 1

        if temp == 0:
            break

        next_state = problem.random_neighbor()
        next_fitness = problem.eval_fitness(next_state)
        delta_e = next_fitness - problem.get_fitness()
        prob = np.exp(delta_e / temp)

        if delta_e > 0 or np.random.uniform() < prob:
            problem.set_state(next_state)
            attempts = 0
        else:
            attempts += 1

        if curve:
            fitness_curve.append(problem.get_fitness())

    best_fitness = problem.get_maximize() * problem.get_fitness()
    best_state = problem.get_state()

    if curve:
        return best_state, best_fitness, np.asarray(fitness_curve)

    return best_state, best_fitness


def genetic_alg(
    problem: DiscreteOpt | ContinuousOpt | TSPOpt,
    pop_size: int = 200,
    mutation_prob: float = 0.1,
    max_attempts: int = 10,
    max_iters: int | float = np.inf,
    curve: bool = False,
    random_state: int = None,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, np.ndarray]:
    """Use a standard genetic algorithm to find the optimum for a given optimization problem.

    Parameters
    ----------
    problem: DiscreteOpt | ContinuousOpt | TSPOpt
        Object containing fitness function optimization problem to be solved.
    pop_size: int, default: 200
        Size of population to be used in genetic algorithm.
    mutation_prob: float, default: 0.1
        Probability of a mutation at each element of the state vector during reproduction, expressed as a value between 0 and 1.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better state at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used by np.random.seed();
        otherwise, the random seed is not set.

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: array
        Numpy array of arrays containing the fitness of the entire population at every iteration.
        Only returned if input argument `curve` is `True`.
    """
    if pop_size < 0:
        raise ValueError("pop_size must be a positive integer.")
    if not isinstance(pop_size, int):
        if pop_size.is_integer():
            pop_size = int(pop_size)
        else:
            raise ValueError("pop_size must be a positive integer.")

    if not (0 <= mutation_prob <= 1):
        raise ValueError("mutation_prob must be between 0 and 1.")

    if not isinstance(max_attempts, int) or max_attempts < 0:
        raise ValueError("max_attempts must be a positive integer.")
    if (not isinstance(max_iters, int) and max_iters != np.inf) or max_iters < 0:
        raise ValueError("max_iters must be a positive integer or np.inf.")

    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    fitness_curve = [] if curve else None
    problem.reset()
    problem.random_pop(pop_size)
    attempts = 0
    iters = 0

    while attempts < max_attempts and iters < max_iters:
        iters += 1
        problem.eval_mate_probs()

        selected = np.random.choice(
            pop_size, size=(pop_size, 2), p=problem.get_mate_probs()
        )
        parents = problem.get_population()[selected]
        children = np.array(
            [problem.reproduce(p[0], p[1], mutation_prob) for p in parents]
        )

        problem.set_population(children)

        next_state = problem.best_child()
        next_fitness = problem.eval_fitness(next_state)

        current_fitness = problem.get_fitness()
        if next_fitness > current_fitness:
            problem.set_state(next_state)
            attempts = 0
        else:
            attempts += 1

        if curve:
            fitness_curve.append(current_fitness)

    best_fitness = problem.get_maximize() * problem.get_fitness()
    best_state = problem.get_state()

    if curve:
        return best_state, best_fitness, np.asarray(fitness_curve)

    return best_state, best_fitness


def mimic(
    problem: DiscreteOpt | ContinuousOpt | TSPOpt,
    pop_size: int = 200,
    keep_pct: float = 0.2,
    max_attempts: int = 10,
    max_iters: int | float = np.inf,
    curve: bool = False,
    random_state: int = None,
    fast_mimic: bool = False,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, np.ndarray]:
    """Use MIMIC to find the optimum for a given optimization problem.

    Parameters
    ----------
    problem: DiscreteOpt | ContinuousOpt | TSPOpt
        Object containing fitness function optimization problem to be solved.
    pop_size: int, default: 200
        Size of population to be used in algorithm.
    keep_pct: float, default: 0.2
        Proportion of samples to keep at each iteration of the algorithm, expressed as a value between 0 and 1.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better neighbor at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used by np.random.seed();
        otherwise, the random seed is not set.
    fast_mimic: bool, default: False
        Activate fast mimic mode to compute the mutual information in vectorized form.
        Faster speed but requires more memory.

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: array
        Numpy array containing the fitness at every iteration. Only returned if input argument `curve` is `True`.
    """
    if problem.get_prob_type() == "continuous":
        raise ValueError("problem type must be discrete or tsp.")

    if pop_size < 0:
        raise ValueError("pop_size must be a positive integer.")
    if not isinstance(pop_size, int):
        if pop_size.is_integer():
            pop_size = int(pop_size)
        else:
            raise ValueError("pop_size must be a positive integer.")

    if not (0 <= keep_pct <= 1):
        raise ValueError("keep_pct must be between 0 and 1.")

    if not isinstance(max_attempts, int) or max_attempts < 0:
        raise ValueError("max_attempts must be a positive integer.")
    if (not isinstance(max_iters, int) and max_iters != np.inf) or max_iters < 0:
        raise ValueError("max_iters must be a positive integer or np.inf.")

    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    fitness_curve = []
    if not isinstance(fast_mimic, bool):
        raise ValueError("fast_mimic mode must be a boolean.")
    else:
        problem.mimic_speed = fast_mimic

    problem.reset()
    problem.random_pop(pop_size)
    attempts = 0
    iters = 0

    while attempts < max_attempts and iters < max_iters:
        iters += 1
        problem.find_top_pct(keep_pct)
        problem.eval_node_probs()
        new_sample = problem.sample_pop(pop_size)
        problem.set_population(new_sample)
        next_state = problem.best_child()
        next_fitness = problem.eval_fitness(next_state)

        current_fitness = problem.get_fitness()
        if next_fitness > current_fitness:
            problem.set_state(next_state)
            attempts = 0
        else:
            attempts += 1

        if curve:
            fitness_curve.append(current_fitness)

    best_fitness = problem.get_maximize() * problem.get_fitness()
    best_state = problem.get_state().astype(int)

    if curve:
        return best_state, best_fitness, np.asarray(fitness_curve)

    return best_state, best_fitness
