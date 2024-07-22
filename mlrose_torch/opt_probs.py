"""Classes for defining optimization problem objects."""

# Author: Genevieve Hayes
# License: BSD 3 clause

import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_tree

from .fitness import TravellingSales


class OptProb:
    """Base class for optimisation problems.

    Parameters
    ----------
    length: int
        Number of elements in state vector.
    fitness_fn: fitness function object
        Object to implement fitness function for optimization.
    maximize: bool, default: True
        Whether to maximize the fitness function.
        Set :code:`False` for minimization problem.
    """

    def __init__(self, length: int, fitness_fn: any, maximize: bool = True) -> None:
        if length < 0:
            raise Exception("""length must be a positive integer.""")
        elif not isinstance(length, int):
            if length.is_integer():
                self.length = int(length)
            else:
                raise Exception("""length must be a positive integer.""")
        else:
            self.length = length

        self.state: np.ndarray = np.zeros(self.length)
        self.neighbors: np.ndarray = np.array([])
        self.fitness_fn = fitness_fn
        self.fitness: float = 0
        self.population: np.ndarray = np.array([])
        self.pop_fitness: np.ndarray = np.array([])
        self.mate_probs: np.ndarray = np.array([])
        self.maximize: float = 1.0 if maximize else -1.0

    def best_child(self) -> np.ndarray:
        """Return the best state in the current population."""
        return self.population[np.argmax(self.pop_fitness)]

    def best_neighbor(self) -> np.ndarray:
        """Return the best neighbor of current state."""
        fitness_list = [self.eval_fitness(neigh) for neigh in self.neighbors]
        return self.neighbors[np.argmax(fitness_list)]

    def eval_fitness(self, state: np.ndarray) -> float:
        """Evaluate the fitness of a state vector."""
        if len(state) != self.length:
            raise Exception("state length must match problem length")
        return self.maximize * self.fitness_fn.evaluate(state)

    def eval_mate_probs(self) -> None:
        """Calculate the probability of each member of the population reproducing."""
        pop_fitness = np.copy(self.pop_fitness)
        pop_fitness[pop_fitness == -1.0 * np.inf] = 0

        total_fitness = np.sum(pop_fitness)
        if total_fitness == 0:
            self.mate_probs = np.ones(len(pop_fitness)) / len(pop_fitness)
        else:
            self.mate_probs = pop_fitness / total_fitness

    def get_fitness(self) -> float:
        """Return the fitness of the current state vector."""
        return self.fitness

    def get_length(self) -> int:
        """Return the state vector length."""
        return self.length

    def get_mate_probs(self) -> np.ndarray:
        """Return the population mate probabilities."""
        return self.mate_probs

    def get_maximize(self) -> float:
        """Return the maximization multiplier."""
        return self.maximize

    def get_pop_fitness(self) -> np.ndarray:
        """Return the current population fitness array."""
        return self.pop_fitness

    def get_population(self) -> np.ndarray:
        return self.population

    def get_state(self) -> np.ndarray:
        """Return the current state vector."""
        return self.state

    def set_population(self, new_population: np.ndarray) -> None:
        """Change the current population to a specified new population and get the fitness of all members."""
        self.population = new_population
        self.pop_fitness = np.array([self.eval_fitness(ind) for ind in new_population])

    def set_state(self, new_state: np.ndarray) -> None:
        """Change the current state vector to a specified value and get its fitness."""
        if len(new_state) != self.length:
            raise Exception("new_state length must match problem length")
        self.state = new_state
        self.fitness = self.eval_fitness(self.state)


class DiscreteOpt(OptProb):
    def __init__(
        self, length: int, fitness_fn: object, maximize: bool = True, max_val: int = 2
    ) -> None:
        super().__init__(length, fitness_fn, maximize)

        if self.fitness_fn.get_prob_type() == "continuous":
            raise Exception(
                "fitness_fn must have problem type 'discrete', 'either' or 'tsp'. "
                "Define problem as ContinuousOpt problem or use alternative fitness function."
            )

        if max_val < 0:
            raise Exception("max_val must be a positive integer.")
        elif not isinstance(max_val, int):
            if max_val.is_integer():
                self.max_val = int(max_val)
            else:
                raise Exception("max_val must be a positive integer.")
        else:
            self.max_val = max_val

        self.keep_sample: np.ndarray = np.array([])
        self.node_probs: np.ndarray = np.zeros(
            [self.length, self.max_val, self.max_val]
        )
        self.parent_nodes: np.ndarray = np.array([])
        self.sample_order: np.ndarray = np.array([])
        self.prob_type: str = "discrete"
        self.mimic_speed: bool = False

    def eval_node_probs(self) -> None:
        """Update probability density estimates."""
        if not self.mimic_speed:
            mutual_info = np.zeros((self.length, self.length))
            for i in range(self.length - 1):
                for j in range(i + 1, self.length):
                    mutual_info[i, j] = -1 * mutual_info_score(
                        self.keep_sample[:, i], self.keep_sample[:, j]
                    )
        else:
            np.seterr(divide="ignore", invalid="ignore")
            len_sample_kept = self.keep_sample.shape[0]
            len_prob = self.keep_sample.shape[1]

            permuted_rows = np.repeat(self.keep_sample, self.length).reshape(
                (len_sample_kept, len_prob * len_prob)
            )
            duplicated_rows = np.hstack([self.keep_sample] * len_prob)

            mutual_info_vectorized = np.zeros([self.length * self.length])
            cluster_U = {}
            cluster_V = {}
            cluster_U_sum = {}
            cluster_V_sum = {}
            for i in range(self.max_val):
                cluster_U[i] = duplicated_rows == i
                cluster_V[i] = permuted_rows == i
                cluster_U_sum[i] = np.sum(duplicated_rows == i, axis=0)
                cluster_V_sum[i] = np.sum(permuted_rows == i, axis=0)

            for i in range(self.max_val):
                for j in range(self.max_val):
                    MI_first_term = (
                        np.sum(cluster_U[i] * cluster_V[j], axis=0) / len_sample_kept
                    )
                    UV_length = cluster_U_sum[i] * cluster_V_sum[j]
                    MI_second_term = (
                        np.log(MI_first_term)
                        - np.log(UV_length)
                        + np.log(len_sample_kept)
                    )

                    MI_second_term[np.isnan(MI_second_term)] = 0
                    MI_second_term[np.isneginf(MI_second_term)] = 0

                    mutual_info_vectorized += MI_first_term * MI_second_term

            mutual_info_full = -1 * mutual_info_vectorized.reshape(
                (self.length, self.length)
            )
            mutual_info = np.triu(mutual_info_full, k=1)

        mst = minimum_spanning_tree(csr_matrix(mutual_info))
        dft = depth_first_tree(csr_matrix(mst.toarray()), 0, directed=False).toarray()
        dft = np.round(dft, 10)
        parent = np.argmin(dft[:, 1:], axis=0)
        probs = np.zeros([self.length, self.max_val, self.max_val])

        probs[0] = np.histogram(
            self.keep_sample[:, 0], np.arange(self.max_val + 1), density=True
        )[0]
        for i in range(1, self.length):
            for j in range(self.max_val):
                subset = self.keep_sample[
                    np.where(self.keep_sample[:, parent[i - 1]] == j)[0]
                ]
                if not len(subset):
                    probs[i, j] = 1 / self.max_val
                else:
                    probs[i, j] = np.histogram(
                        subset[:, i], np.arange(self.max_val + 1), density=True
                    )[0]

        self.node_probs = probs
        self.parent_nodes = parent

    def find_neighbors(self) -> None:
        """Find all neighbors of the current state."""
        self.neighbors = []
        state_copy = np.copy(self.state)
        if self.max_val == 2:
            for i in range(self.length):
                neighbor = state_copy.copy()
                neighbor[i] = np.abs(neighbor[i] - 1)
                self.neighbors.append(neighbor)
        else:
            vals = np.arange(self.max_val)
            for i in range(self.length):
                current_val = state_copy[i]
                for j in vals[vals != current_val]:
                    neighbor = state_copy.copy()
                    neighbor[i] = j
                    self.neighbors.append(neighbor)

    def find_sample_order(self) -> None:
        """Determine order in which to generate sample vector elements."""
        sample_order = []
        last = [0]
        parent = self.parent_nodes

        while len(sample_order) < self.length:
            idxs = []

            if not last:
                remaining_indices = set(np.arange(self.length)) - set(sample_order)
                idxs.append(np.random.choice(list(remaining_indices)))
            else:
                for i in last:
                    idxs.extend(np.where(parent == i)[0] + 1)

            sample_order.extend(last)
            last = idxs

        self.sample_order = sample_order

    def find_top_pct(self, keep_pct: float) -> None:
        """Select samples with fitness in the top keep_pct percentile."""
        if not (0 <= keep_pct <= 1):
            raise Exception("keep_pct must be between 0 and 1.")
        theta = np.percentile(self.pop_fitness, 100 * (1 - keep_pct))
        keep_idxs = np.where(self.pop_fitness >= theta)[0]
        self.keep_sample = self.population[keep_idxs]

    def get_keep_sample(self) -> np.ndarray:
        return self.keep_sample

    def get_prob_type(self) -> str:
        return self.prob_type

    def random(self) -> np.ndarray:
        """Return a random state vector."""
        return np.random.randint(0, self.max_val, self.length)

    def random_neighbor(self) -> np.ndarray:
        """Return random neighbor of current state vector."""
        neighbor = np.copy(self.state)
        i = np.random.randint(0, self.length)
        if self.max_val == 2:
            neighbor[i] = np.abs(neighbor[i] - 1)
        else:
            vals = list(np.arange(self.max_val))
            vals.remove(neighbor[i])
            neighbor[i] = vals[np.random.randint(0, self.max_val - 1)]
        return neighbor

    def random_pop(self, pop_size: int) -> None:
        """Create a population of random state vectors."""
        if not isinstance(pop_size, int) or pop_size <= 0:
            raise Exception("pop_size must be a positive integer.")
        population = [self.random() for _ in range(pop_size)]
        pop_fitness = [self.eval_fitness(state) for state in population]
        self.population = np.array(population)
        self.pop_fitness = np.array(pop_fitness)

    def reproduce(
        self, parent_1: np.ndarray, parent_2: np.ndarray, mutation_prob: float = 0.1
    ) -> np.ndarray:
        """Create child state vector from two parent state vectors."""
        if len(parent_1) != self.length or len(parent_2) != self.length:
            raise Exception("Lengths of parents must match problem length")
        if not (0 <= mutation_prob <= 1):
            raise Exception("mutation_prob must be between 0 and 1.")
        if self.length > 1:
            _n = np.random.randint(self.length - 1)
            child = np.concatenate([parent_1[: _n + 1], parent_2[_n + 1 :]])
        else:
            child = np.copy(parent_1 if np.random.randint(2) == 0 else parent_2)

        rand = np.random.uniform(size=self.length)
        mutate = np.where(rand < mutation_prob)[0]
        if self.max_val == 2:
            child[mutate] = np.abs(child[mutate] - 1)
        else:
            for i in mutate:
                vals = list(np.arange(self.max_val))
                vals.remove(child[i])
                child[i] = vals[np.random.randint(0, self.max_val - 1)]
        return child

    def reset(self) -> None:
        """Set the current state vector to a random value and get its fitness."""
        self.state = self.random()
        self.fitness = self.eval_fitness(self.state)

    def sample_pop(self, sample_size: int) -> np.ndarray:
        """Generate new sample from probability density."""
        if not isinstance(sample_size, int) or sample_size <= 0:
            raise Exception("sample_size must be a positive integer.")
        new_sample = np.zeros([sample_size, self.length])
        new_sample[:, 0] = np.random.choice(
            self.max_val, sample_size, p=self.node_probs[0, 0]
        )
        self.find_sample_order()
        sample_order = self.sample_order[1:]

        for i in sample_order:
            par_ind = self.parent_nodes[i - 1]
            for j in range(self.max_val):
                idxs = np.where(new_sample[:, par_ind] == j)[0]
                new_sample[idxs, i] = np.random.choice(
                    self.max_val, len(idxs), p=self.node_probs[i, j]
                )
        return new_sample


class ContinuousOpt(OptProb):
    """Class for defining continuous-state optimisation problems.

    Parameters
    ----------
    length: int
        Number of elements in state vector.

    fitness_fn: fitness function object
        Object to implement fitness function for optimization.

    maximize: bool, default: True
        Whether to maximize the fitness function.
        Set :code:`False` for minimization problem.

    min_val: float, default: 0
        Minimum value that each element of the state vector can take.

    max_val: float, default: 1
        Maximum value that each element of the state vector can take.

    step: float, default: 0.1
        Step size used in determining neighbors of current state.
    """

    def __init__(
        self,
        length: int,
        fitness_fn: object,
        maximize: bool = True,
        min_val: float = 0,
        max_val: float = 1,
        step: float = 0.1,
    ) -> None:
        super().__init__(length, fitness_fn, maximize=maximize)

        prob_type = self.fitness_fn.get_prob_type()
        if prob_type not in ["continuous", "either"]:
            raise Exception(
                "fitness_fn must have problem type 'continuous' or 'either'. "
                "Define problem as DiscreteOpt or use alternative fitness function."
            )

        if max_val <= min_val:
            raise Exception("max_val must be greater than min_val.")
        if step <= 0:
            raise Exception("step size must be positive.")
        if (max_val - min_val) < step:
            raise Exception("step size must be less than (max_val - min_val).")

        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.prob_type: str = "continuous"

    def calculate_updates(self) -> np.ndarray:
        """Calculate gradient descent updates."""
        return self.fitness_fn.calculate_updates()

    def find_neighbors(self) -> None:
        """Find all neighbors of the current state."""
        self.neighbors = []

        for i in range(self.length):
            for j in [-1, 1]:
                neighbor = np.copy(self.state)
                neighbor[i] = np.clip(
                    neighbor[i] + j * self.step, self.min_val, self.max_val
                )
                if not np.array_equal(neighbor, self.state):
                    self.neighbors.append(neighbor)

    def get_prob_type(self) -> str:
        return self.prob_type

    def random(self) -> np.ndarray:
        """Return a random state vector."""
        return np.random.uniform(self.min_val, self.max_val, self.length)

    def random_neighbor(self) -> np.ndarray:
        """Return random neighbor of current state vector."""
        while True:
            neighbor = np.copy(self.state)
            i = np.random.randint(self.length)
            neighbor[i] += self.step * np.random.choice([-1, 1])
            neighbor[i] = np.clip(neighbor[i], self.min_val, self.max_val)
            if not np.array_equal(neighbor, self.state):
                break
        return neighbor

    def random_pop(self, pop_size: int) -> None:
        """Create a population of random state vectors."""
        if not isinstance(pop_size, int) or pop_size <= 0:
            raise Exception("pop_size must be a positive integer.")

        population = np.random.uniform(
            self.min_val, self.max_val, (pop_size, self.length)
        )
        pop_fitness = np.array([self.eval_fitness(state) for state in population])

        self.population = population
        self.pop_fitness = pop_fitness

    def reproduce(
        self, parent_1: np.ndarray, parent_2: np.ndarray, mutation_prob: float = 0.1
    ) -> np.ndarray:
        """Create child state vector from two parent state vectors."""
        if len(parent_1) != self.length or len(parent_2) != self.length:
            raise Exception("Lengths of parents must match problem length")

        if not (0 <= mutation_prob <= 1):
            raise Exception("mutation_prob must be between 0 and 1.")

        # Reproduce parents
        if self.length > 1:
            _n = np.random.randint(self.length - 1)
            child = np.zeros(self.length, dtype=float)
            child[: _n + 1] = parent_1[: _n + 1]
            child[_n + 1 :] = parent_2[_n + 1 :]
        elif np.random.randint(2) == 0:
            child = np.copy(parent_1)
        else:
            child = np.copy(parent_2)

        # Mutate child
        if mutation_prob > 0:
            rand = np.random.uniform(size=self.length)
            mutate = np.where(rand < mutation_prob)[0]

            for i in mutate:
                child[i] = np.random.uniform(self.min_val, self.max_val)

        return child

    def reset(self) -> None:
        """Set the current state vector to a random value and get its fitness."""
        self.state = self.random()
        self.fitness = self.eval_fitness(self.state)

    def update_state(self, updates: np.ndarray) -> np.ndarray:
        """Update current state given a vector of updates."""
        if len(updates) != self.length:
            raise Exception("Length of updates must match problem length")
        return np.clip(self.state + updates, self.min_val, self.max_val)


class TSPOpt(DiscreteOpt):
    """Class for defining travelling salesperson optimisation problems.

    Parameters
    ----------
    length: int
        Number of elements in state vector. Must equal number of nodes in the
        tour.

    fitness_fn: fitness function object, default: None
        Object to implement fitness function for optimization. If :code:`None`,
        then :code:`TravellingSales(coords=coords, distances=distances)` is
        used by default.

    maximize: bool, default: False
        Whether to maximize the fitness function.
        Set :code:`False` for minimization problem.

    coords: list of pairs, default: None
        Ordered list of the (x, y) co-ordinates of all nodes. This assumes
        that travel between all pairs of nodes is possible. If this is not the
        case, then use distances instead. This argument is ignored if
        fitness_fn is not :code:`None`.

    distances: list of triples, default: None
        List giving the distances, d, between all pairs of nodes, u and v, for
        which travel is possible, with each list item in the form (u, v, d).
        Order of the nodes does not matter, so (u, v, d) and (v, u, d) are
        considered to be the same. If a pair is missing from the list, it is
        assumed that travel between the two nodes is not possible. This
        argument is ignored if fitness_fn or coords is not :code:`None`.
    """

    def __init__(
        self,
        length: int,
        fitness_fn: object = None,
        maximize: bool = False,
        coords: list[tuple] = None,
        distances: list[tuple] = None,
    ) -> None:
        if (fitness_fn is None) and (coords is None) and (distances is None):
            raise Exception(
                "At least one of fitness_fn, coords and distances must be specified."
            )
        elif fitness_fn is None:
            fitness_fn = TravellingSales(coords=coords, distances=distances)

        super().__init__(length, fitness_fn, maximize, max_val=length)

        if self.fitness_fn.get_prob_type() != "tsp":
            raise Exception("fitness_fn must have problem type 'tsp'.")

        self.prob_type: str = "tsp"

    def adjust_probs(self, probs: np.ndarray) -> np.ndarray:
        """Normalize a vector of probabilities so that the vector sums to 1."""
        if np.sum(probs) == 0:
            return np.zeros(np.shape(probs))
        return probs / np.sum(probs)

    def find_neighbors(self) -> None:
        """Find all neighbors of the current state."""
        self.neighbors = []
        state = self.state
        for node1 in range(self.length - 1):
            for node2 in range(node1 + 1, self.length):
                neighbor = np.copy(state)
                neighbor[node1], neighbor[node2] = state[node2], state[node1]
                self.neighbors.append(neighbor)

    def random(self) -> np.ndarray:
        """Return a random state vector."""
        return np.random.permutation(self.length)

    def random_mimic(self) -> np.ndarray:
        """Generate single MIMIC sample from probability density."""
        remaining = list(np.arange(self.length))
        state = np.zeros(self.length, dtype=np.int8)
        node_probs = np.copy(self.node_probs)

        state[0] = np.random.choice(self.length, p=node_probs[0, 0])
        remaining.remove(state[0])
        node_probs[:, :, state[0]] = 0

        self.find_sample_order()
        sample_order = self.sample_order[1:]

        for i in sample_order:
            par_ind = self.parent_nodes[i - 1]
            par_value = state[par_ind]
            probs = node_probs[i, par_value]

            if np.sum(probs) == 0:
                next_node = np.random.choice(remaining)
            else:
                adj_probs = self.adjust_probs(probs)
                next_node = np.random.choice(self.length, p=adj_probs)

            state[i] = next_node
            remaining.remove(next_node)
            node_probs[:, :, next_node] = 0

        return state

    def random_neighbor(self) -> np.ndarray:
        """Return random neighbor of current state vector."""
        neighbor = np.copy(self.state)
        node1, node2 = np.random.choice(np.arange(self.length), size=2, replace=False)
        neighbor[node1], neighbor[node2] = neighbor[node2], neighbor[node1]
        return neighbor

    def reproduce(
        self, parent_1: np.ndarray, parent_2: np.ndarray, mutation_prob: float = 0.1
    ) -> np.ndarray:
        """Create child state vector from two parent state vectors."""
        if len(parent_1) != self.length or len(parent_2) != self.length:
            raise Exception("Lengths of parents must match problem length")
        if not (0 <= mutation_prob <= 1):
            raise Exception("mutation_prob must be between 0 and 1.")

        if self.length > 1:
            _n = np.random.randint(self.length - 1)
            child = np.zeros(self.length, dtype=np.int8)
            child[: _n + 1] = parent_1[0 : _n + 1]
            unvisited = [node for node in parent_2 if node not in parent_1[: _n + 1]]
            child[_n + 1 :] = unvisited
        else:
            child = np.copy(parent_1 if np.random.randint(2) == 0 else parent_2)

        rand = np.random.uniform(size=self.length)
        mutate = np.where(rand < mutation_prob)[0]

        if len(mutate) > 0:
            mutate_perm = np.random.permutation(mutate)
            child[mutate] = child[mutate_perm]

        return child

    def sample_pop(self, sample_size: int) -> np.ndarray:
        """Generate new sample from probability density."""
        if sample_size <= 0:
            raise Exception("sample_size must be a positive integer.")
        elif not isinstance(sample_size, int):
            if sample_size.is_integer():
                sample_size = int(sample_size)
            else:
                raise Exception("sample_size must be a positive integer.")

        self.find_sample_order()
        return np.array([self.random_mimic() for _ in range(sample_size)])
