#!/bin/bash

echo "Running tests on test_activation.py"
python tests/test_activation.py

echo "Running tests on test_decay.py"
python tests/test_decay.py

echo "Running tests on test_fitness.py"
python tests/test_fitness.py

echo "Running tests on test_algorithms.py"
python tests/test_algorithms.py

echo "Running tests on test_opt_probs.py"
python tests/test_opt_probs.py

echo "Running tests on test_neural.py"
python tests/test_neural.py

echo "Finished all tests"
