"""Classes for defining neural network weight optimization problems."""

# Author: Genevieve Hayes
# License: BSD 3 clause

from abc import ABCMeta
from abc import abstractmethod

import six
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import mean_squared_error, log_loss

from .decay import GeomDecay
from .opt_probs import ContinuousOpt
from .activation import identity, relu, sigmoid, softmax, tanh
from .algorithms import random_hill_climb, simulated_annealing, genetic_alg


def flatten_weights(weights: list[np.ndarray]) -> np.ndarray:
    """Flatten list of weights arrays into a 1D array.

    Parameters
    ----------
    weights: list of arrays
        List of 2D arrays for flattening.

    Returns
    -------
    flat_weights: np.ndarray
        1D weights array.
    """
    return np.concatenate([w.flatten() for w in weights])


def unflatten_weights(
    flat_weights: np.ndarray, node_list: list[int]
) -> list[np.ndarray]:
    """Convert 1D weights array into list of 2D arrays.

    Parameters
    ----------
    flat_weights: np.ndarray
        1D weights array.

    node_list: list
        List giving the number of nodes in each layer of the network,
        including the input and output layers.

    Returns
    -------
    weights: list of arrays
        List of 2D arrays created from flat_weights.
    """
    weights = []
    start = 0
    for i in range(len(node_list) - 1):
        end = start + node_list[i] * node_list[i + 1]
        weights.append(
            flat_weights[start:end].reshape((node_list[i], node_list[i + 1]))
        )
        start = end
    return weights


def gradient_descent(
    problem: any,
    max_attempts: int = 10,
    max_iters: int = np.inf,
    init_state: np.ndarray = None,
    curve: bool = False,
    random_state: any = None,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, np.ndarray]:
    """Use gradient_descent to find the optimal neural network weights.

    Parameters
    ----------
    problem: optimization object
        Object containing optimization problem to be solved.

    max_attempts: int, default: 10
        Maximum number of attempts to find a better state at each step.

    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.

    init_state: np.ndarray, default: None
        Numpy array containing starting state for algorithm.
        If None, then a random state is used.

    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used
        by np.random.seed(); otherwise, the random seed is not set.

    curve: bool, default: False
        Boolean to keep fitness values for a curve.
        If :code:`False`, then no curve is stored.
        If :code:`True`, then a history of fitness values is provided as a
        third return value.

    Returns
    -------
    best_state: np.ndarray
        Numpy array containing state that optimizes fitness function.

    best_fitness: float
        Value of fitness function at best state.

    fitness_curve: np.ndarray
        Numpy array containing the fitness at every iteration.
        Only returned if input argument :code:`curve` is :code:`True`.
    """
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
    best_fitness = problem.get_maximize() * problem.get_fitness()
    best_state = problem.get_state()

    while attempts < max_attempts and iters < max_iters:
        iters += 1
        updates = flatten_weights(problem.calculate_updates())
        next_state = problem.update_state(updates)
        next_fitness = problem.eval_fitness(next_state)

        if next_fitness > problem.get_fitness():
            attempts = 0
        else:
            attempts += 1

        if next_fitness > problem.get_maximize() * best_fitness:
            best_fitness = problem.get_maximize() * next_fitness
            best_state = next_state

        if curve:
            fitness_curve.append(problem.get_fitness())

        problem.set_state(next_state)

    if curve:
        return best_state, best_fitness, np.asarray(fitness_curve)

    return best_state, best_fitness


class NetworkWeights:
    """Fitness function for neural network weights optimization problem.

    Parameters
    ----------
    X: np.ndarray
        Numpy array containing feature dataset with each row representing a
        single observation.

    y: np.ndarray
        Numpy array containing true values of data labels.
        Length must be same as length of X.

    node_list: list of ints
        Number of nodes in each layer, including the input and output layers.

    activation: callable
        Activation function for each of the hidden layers with the signature
        :code:`activation(x, deriv)`, where setting deriv is a boolean that
        determines whether to return the activation function or its derivative.

    bias: bool, default: True
        Whether a bias term is included in the network.

    is_classifier: bool, default: True
        Whether the network is for classification or regression. Set True for
        classification and False for regression.

    learning_rate: float, default: 0.1
        Learning rate for gradient descent.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        node_list: list[int],
        activation: any,
        bias: bool = True,
        is_classifier: bool = True,
        learning_rate: float = 0.1,
    ):
        y = np.array(y)
        if len(np.shape(y)) == 1:
            y = y.reshape((len(y), 1))
        if not np.shape(X)[0] == np.shape(y)[0]:
            raise ValueError("The length of X and y must be equal.")
        if len(node_list) < 2:
            raise ValueError("node_list must contain at least 2 elements.")
        if not np.shape(X)[1] == (node_list[0] - bias):
            raise ValueError(
                "The number of columns in X must equal %d" % (node_list[0] - bias)
            )
        if not np.shape(y)[1] == node_list[-1]:
            raise ValueError(
                "The number of columns in y must equal %d" % (node_list[-1],)
            )
        if not isinstance(bias, bool):
            raise ValueError("bias must be True or False.")
        if not isinstance(is_classifier, bool):
            raise ValueError("is_classifier must be True or False.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")

        self.X = X
        self.y_true = y
        self.node_list = node_list
        self.activation = activation
        self.bias = bias
        self.is_classifier = is_classifier
        self.learning_rate = learning_rate
        self.loss = log_loss if is_classifier else mean_squared_error
        self.output_activation = (
            sigmoid
            if is_classifier and np.shape(self.y_true)[1] == 1
            else softmax if is_classifier else identity
        )
        self.inputs_list = []
        self.y_pred = y
        self.weights = []
        self.prob_type = "continuous"
        self.nodes = sum(
            node_list[i] * node_list[i + 1] for i in range(len(node_list) - 1)
        )

    def evaluate(self, state: np.ndarray) -> float:
        """Evaluate the fitness of a state."""
        if not len(state) == self.nodes:
            raise ValueError("state must have length %d" % (self.nodes,))
        self.inputs_list = []
        self.weights = unflatten_weights(np.asarray(state), self.node_list)

        inputs = (
            np.hstack((self.X, np.ones((np.shape(self.X)[0], 1))))
            if self.bias
            else self.X
        )

        for i in range(len(self.weights)):
            outputs = np.dot(inputs, self.weights[i])
            self.inputs_list.append(inputs)
            inputs = (
                self.activation(outputs)
                if i < len(self.weights) - 1
                else self.output_activation(outputs)
            )
            self.y_pred = inputs

        return self.loss(self.y_true, self.y_pred)

    def get_output_activation(self) -> callable:
        """Return the activation function for the output layer."""
        return self.output_activation

    def get_prob_type(self) -> str:
        """Return the problem type ('discrete', 'continuous', 'tsp', or 'either')."""
        return self.prob_type

    def calculate_updates(self) -> list:
        """Calculate gradient descent updates.

        Returns
        -------
        updates_list: list
            List of back propagation weight updates.
        """
        delta_list = []
        updates_list = []

        for i in range(len(self.inputs_list) - 1, -1, -1):
            if i == len(self.inputs_list) - 1:
                delta = self.y_pred - self.y_true
            else:
                delta = np.dot(delta_list[-1], self.weights[i + 1].T) * self.activation(
                    self.inputs_list[i + 1], deriv=True
                )

            delta_list.append(delta)
            updates = -self.learning_rate * np.dot(self.inputs_list[i].T, delta)
            updates_list.append(updates)

        return updates_list[::-1]


class BaseNeuralNetwork(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for neural networks.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(
        self,
        hidden_nodes: any = None,
        activation: str = "relu",
        algorithm: str = "random_hill_climb",
        max_iters: int = 100,
        bias: bool = True,
        is_classifier: bool = True,
        learning_rate: float = 0.1,
        early_stopping: bool = False,
        clip_max: float = 1e10,
        restarts: int = 0,
        schedule: GeomDecay = GeomDecay(),
        pop_size: int = 200,
        mutation_prob: float = 0.1,
        max_attempts: int = 10,
        random_state: any = None,
        curve: bool = False,
    ):
        self.hidden_nodes = hidden_nodes if hidden_nodes is not None else []
        self.activation_dict = {
            "identity": identity,
            "relu": relu,
            "sigmoid": sigmoid,
            "tanh": tanh,
        }
        self.activation = activation
        self.algorithm = algorithm
        self.max_iters = max_iters
        self.bias = bias
        self.is_classifier = is_classifier
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.clip_max = clip_max
        self.restarts = restarts
        self.schedule = schedule if schedule is not None else GeomDecay()
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.max_attempts = max_attempts
        self.random_state = random_state
        self.curve = curve
        self.node_list = []
        self.fitted_weights = []
        self.loss = np.inf
        self.output_activation = None
        self.predicted_probs = []
        self.fitness_curve = []

    def _validate(self):
        if (
            not isinstance(self.max_iters, int)
            and self.max_iters != np.inf
            and not self.max_iters.is_integer()
        ) or (self.max_iters < 0):
            raise ValueError("max_iters must be a positive integer.")
        if not isinstance(self.bias, bool):
            raise ValueError("bias must be True or False.")
        if not isinstance(self.is_classifier, bool):
            raise ValueError("is_classifier must be True or False.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")
        if not isinstance(self.early_stopping, bool):
            raise ValueError("early_stopping must be True or False.")
        if self.clip_max <= 0:
            raise ValueError("clip_max must be greater than 0.")
        if (
            not isinstance(self.max_attempts, int)
            and not self.max_attempts.is_integer()
        ) or (self.max_attempts < 0):
            raise ValueError("max_attempts must be a positive integer.")
        if self.pop_size < 0:
            raise ValueError("pop_size must be a positive integer.")
        elif not isinstance(self.pop_size, int):
            if self.pop_size.is_integer():
                self.pop_size = int(self.pop_size)
            else:
                raise ValueError("pop_size must be a positive integer.")
        if (self.mutation_prob < 0) or (self.mutation_prob > 1):
            raise ValueError("mutation_prob must be between 0 and 1.")
        if self.activation not in self.activation_dict.keys():
            raise ValueError(
                "Activation function must be one of: 'identity', 'relu', 'sigmoid' or 'tanh'."
            )
        if self.algorithm not in [
            "random_hill_climb",
            "simulated_annealing",
            "genetic_alg",
            "gradient_descent",
        ]:
            raise ValueError(
                "Algorithm must be one of: 'random_hill_climb', 'simulated_annealing', 'genetic_alg', "
                "'gradient_descent'."
            )

    def fit(self, X: np.ndarray, y: np.ndarray = None, init_weights: np.ndarray = None):
        """Fit neural network to data.

        Parameters
        ----------
        X: np.ndarray
            Numpy array containing feature dataset with each row
            representing a single observation.

        y: np.ndarray
            Numpy array containing data labels. Length must be same as
            length of X.

        init_weights: np.ndarray, default: None
            Numpy array containing starting weights for algorithm.
            If :code:`None`, then a random state is used.
        """
        self._validate()

        # Make sure y is an array and not a list
        y = np.array(y)

        # Convert y to 2D if necessary
        if len(np.shape(y)) == 1:
            y = np.reshape(y, [len(y), 1])

        # Verify X and y are the same length
        if not np.shape(X)[0] == np.shape(y)[0]:
            raise Exception("The length of X and y must be equal.")

        # Determine number of nodes in each layer
        input_nodes = np.shape(X)[1] + self.bias
        output_nodes = np.shape(y)[1]
        node_list = [input_nodes] + self.hidden_nodes + [output_nodes]

        num_nodes = 0

        for i in range(len(node_list) - 1):
            num_nodes += node_list[i] * node_list[i + 1]

        if init_weights is not None and len(init_weights) != num_nodes:
            raise Exception(
                """init_weights must be None or have length %d""" % (num_nodes,)
            )

        # Set random seed
        if isinstance(self.random_state, int) and self.random_state > 0:
            np.random.seed(self.random_state)

        # Initialize optimization problem
        fitness = NetworkWeights(
            X,
            y,
            node_list,
            self.activation_dict[self.activation],
            self.bias,
            self.is_classifier,
            learning_rate=self.learning_rate,
        )

        fitness_curve = []

        problem = ContinuousOpt(
            num_nodes,
            fitness,
            maximize=False,
            min_val=-1 * self.clip_max,
            max_val=self.clip_max,
            step=self.learning_rate,
        )

        if self.algorithm == "random_hill_climb":
            fitted_weights = None
            loss = np.inf

            # Can't use restart feature of random_hill_climb function, since
            # want to keep initial weights in the range -1 to 1.
            for _ in range(self.restarts + 1):
                if init_weights is None:
                    init_weights = np.random.uniform(-1, 1, num_nodes)

                if self.curve:
                    current_weights, current_loss, fitness_curve = random_hill_climb(
                        problem,
                        max_attempts=(
                            self.max_attempts if self.early_stopping else self.max_iters
                        ),
                        max_iters=self.max_iters,
                        restarts=0,
                        init_state=init_weights,
                        curve=self.curve,
                    )
                else:
                    current_weights, current_loss = random_hill_climb(
                        problem,
                        max_attempts=(
                            self.max_attempts if self.early_stopping else self.max_iters
                        ),
                        max_iters=self.max_iters,
                        restarts=0,
                        init_state=init_weights,
                        curve=self.curve,
                    )

                if current_loss < loss:
                    fitted_weights = current_weights
                    loss = current_loss

        elif self.algorithm == "simulated_annealing":
            if init_weights is None:
                init_weights = np.random.uniform(-1, 1, num_nodes)

            if self.curve:
                fitted_weights, loss, fitness_curve = simulated_annealing(
                    problem,
                    schedule=self.schedule,
                    max_attempts=(
                        self.max_attempts if self.early_stopping else self.max_iters
                    ),
                    max_iters=self.max_iters,
                    init_state=init_weights,
                    curve=self.curve,
                )
            else:
                fitted_weights, loss = simulated_annealing(
                    problem,
                    schedule=self.schedule,
                    max_attempts=(
                        self.max_attempts if self.early_stopping else self.max_iters
                    ),
                    max_iters=self.max_iters,
                    init_state=init_weights,
                    curve=self.curve,
                )

        elif self.algorithm == "genetic_alg":
            if self.curve:
                fitted_weights, loss, fitness_curve = genetic_alg(
                    problem,
                    pop_size=self.pop_size,
                    mutation_prob=self.mutation_prob,
                    max_attempts=(
                        self.max_attempts if self.early_stopping else self.max_iters
                    ),
                    max_iters=self.max_iters,
                    curve=self.curve,
                )
            else:
                fitted_weights, loss = genetic_alg(
                    problem,
                    pop_size=self.pop_size,
                    mutation_prob=self.mutation_prob,
                    max_attempts=(
                        self.max_attempts if self.early_stopping else self.max_iters
                    ),
                    max_iters=self.max_iters,
                    curve=self.curve,
                )

        else:  # Gradient descent case
            if init_weights is None:
                init_weights = np.random.uniform(-1, 1, num_nodes)

            if self.curve:
                fitted_weights, loss, fitness_curve = gradient_descent(
                    problem,
                    max_attempts=(
                        self.max_attempts if self.early_stopping else self.max_iters
                    ),
                    max_iters=self.max_iters,
                    curve=self.curve,
                    init_state=init_weights,
                )

            else:
                fitted_weights, loss = gradient_descent(
                    problem,
                    max_attempts=(
                        self.max_attempts if self.early_stopping else self.max_iters
                    ),
                    max_iters=self.max_iters,
                    curve=self.curve,
                    init_state=init_weights,
                )

        # Save fitted weights and node list
        self.node_list = node_list
        self.fitted_weights = fitted_weights
        self.loss = loss
        self.output_activation = fitness.get_output_activation()

        if self.curve:
            self.fitness_curve = fitness_curve

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Use model to predict data labels for given feature array.

        Parameters
        ----------
        X: np.ndarray
            Numpy array containing feature dataset with each row
            representing a single observation.

        Returns
        -------
        y_pred: np.ndarray
            Numpy array containing predicted data labels.
        """
        if not np.shape(X)[1] == (self.node_list[0] - self.bias):
            raise ValueError(
                "The number of columns in X must equal %d"
                % (self.node_list[0] - self.bias)
            )

        weights = unflatten_weights(np.asarray(self.fitted_weights), self.node_list)
        inputs = np.hstack((X, np.ones((np.shape(X)[0], 1)))) if self.bias else X

        for i in range(len(weights)):
            outputs = np.dot(inputs, weights[i])
            inputs = (
                self.activation_dict[self.activation](outputs)
                if i < len(weights) - 1
                else self.output_activation(outputs)
            )

        y_pred = inputs

        if self.is_classifier:
            self.predicted_probs = y_pred
            if self.node_list[-1] == 1:
                y_pred = np.round(y_pred).astype(int)
            else:
                y_pred = np.eye(self.node_list[-1])[np.argmax(y_pred, axis=1)]

        return np.asarray(y_pred)

    def get_params(self, deep: bool = False) -> dict:
        return {
            "hidden_nodes": self.hidden_nodes,
            "max_iters": self.max_iters,
            "bias": self.bias,
            "is_classifier": self.is_classifier,
            "learning_rate": self.learning_rate,
            "early_stopping": self.early_stopping,
            "clip_max": self.clip_max,
            "restarts": self.restarts,
            "schedule": self.schedule,
            "pop_size": self.pop_size,
            "mutation_prob": self.mutation_prob,
        }

    def set_params(self, **in_params: dict[any]) -> None:
        for key, value in in_params.items():
            setattr(self, key, value)


class NeuralNetwork(BaseNeuralNetwork, ClassifierMixin):
    def __init__(
        self,
        hidden_nodes: any = None,
        activation: str = "relu",
        algorithm: str = "random_hill_climb",
        max_iters: int = 100,
        bias: bool = True,
        is_classifier: bool = True,
        learning_rate: float = 0.1,
        early_stopping: bool = False,
        clip_max: float = 1e10,
        restarts: int = 0,
        schedule: any = None,
        pop_size: int = 200,
        mutation_prob: float = 0.1,
        max_attempts: int = 10,
        random_state: any = None,
        curve: bool = False,
    ) -> None:
        super(NeuralNetwork, self).__init__(
            hidden_nodes=hidden_nodes,
            activation=activation,
            algorithm=algorithm,
            max_iters=max_iters,
            bias=bias,
            is_classifier=is_classifier,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            clip_max=clip_max,
            restarts=restarts,
            schedule=schedule if schedule is not None else GeomDecay(),
            pop_size=pop_size,
            mutation_prob=mutation_prob,
            max_attempts=max_attempts,
            random_state=random_state,
            curve=curve,
        )


class LinearRegression(BaseNeuralNetwork, RegressorMixin):
    def __init__(
        self,
        algorithm: str = "random_hill_climb",
        max_iters: int = 100,
        bias: bool = True,
        learning_rate: float = 0.1,
        early_stopping: bool = False,
        clip_max: float = 1e10,
        restarts: int = 0,
        schedule: any = None,
        pop_size: int = 200,
        mutation_prob: float = 0.1,
        max_attempts: int = 10,
        random_state: any = None,
        curve: bool = False,
    ) -> None:
        super(LinearRegression, self).__init__(
            hidden_nodes=[],
            activation="identity",
            algorithm=algorithm,
            max_iters=max_iters,
            bias=bias,
            is_classifier=False,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            clip_max=clip_max,
            restarts=restarts,
            schedule=schedule if schedule is not None else GeomDecay(),
            pop_size=pop_size,
            mutation_prob=mutation_prob,
            max_attempts=max_attempts,
            random_state=random_state,
            curve=curve,
        )


class LogisticRegression(BaseNeuralNetwork, ClassifierMixin):
    def __init__(
        self,
        algorithm: str = "random_hill_climb",
        max_iters: int = 100,
        bias: bool = True,
        learning_rate: float = 0.1,
        early_stopping: bool = False,
        clip_max: float = 1e10,
        restarts: int = 0,
        schedule: any = None,
        pop_size: int = 200,
        mutation_prob: float = 0.1,
        max_attempts: int = 10,
        random_state: any = None,
        curve: bool = False,
    ) -> None:
        super(LogisticRegression, self).__init__(
            hidden_nodes=[],
            activation="sigmoid",
            algorithm=algorithm,
            max_iters=max_iters,
            bias=bias,
            is_classifier=True,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            clip_max=clip_max,
            restarts=restarts,
            schedule=schedule if schedule is not None else GeomDecay(),
            pop_size=pop_size,
            mutation_prob=mutation_prob,
            max_attempts=max_attempts,
            random_state=random_state,
            curve=curve,
        )
