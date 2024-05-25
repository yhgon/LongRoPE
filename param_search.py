from transformers import LlamaForCausalLM
import transformers
import torch
import math
import os
import warnings
from typing import Optional, Tuple, Union, List, Dict
 
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
 
def evolutionary_search(model, data, population_size=64, num_mutations=16, num_crossovers=16, max_iterations=40):
    def initialize_population():
        population = []
        for _ in range(population_size):
            lambda_factors = torch.FloatTensor(model.dim).uniform_(1.0, model.extension_ratio)
            n_hat = np.random.randint(0, model.dim)
            population.append((lambda_factors, n_hat))
        return population

    def evaluate_individual(individual):
        lambda_factors, n_hat = individual
        model.lambda_factors = lambda_factors
        model.n_prime = n_hat
        # Calculate perplexity or other performance metric here
        perplexity = 0  # Placeholder
        return perplexity

    def select_top_k(population, k):
        sorted_population = sorted(population, key=evaluate_individual)
        return sorted_population[:k]

    def mutate(parents):
        children = []
        for parent in parents:
            child_lambda = parent[0].clone()
            child_n_hat = parent[1]
            for i in range(model.dim // 2):
                if np.random.rand() < 0.1:
                    child_lambda[i] *= np.random.uniform(0.8, 1.2)
            if np.random.rand() < 0.1:
                child_n_hat = np.random.randint(0, model.dim)
            children.append((child_lambda, child_n_hat))
        return children

    def crossover(parents):
        children = []
        for _ in range(num_crossovers):
            parent1, parent2 = np.random.choice(parents, 2)
            child_lambda = parent1[0].clone()
            for i in range(model.dim // 2):
                if np.random.rand() < 0.5:
                    child_lambda[i] = parent2[0][i]
            child_n_hat = parent2[1] if np.random.rand() < 0.5 else parent1[1]
            children.append((child_lambda, child_n_hat))
        return children

    population = initialize_population()
    for _ in range(max_iterations):
        evaluated_population = [(individual, evaluate_individual(individual)) for individual in population]
        parents = select_top_k(evaluated_population, population_size // 2)
        population = parents + mutate(parents) + crossover(parents)
    
    best_individual = min(population, key=evaluate_individual)
    return best_individual


