"""
binary_ga.py
------------
Standard binary Genetic Algorithm for wrapper-based feature selection --
baseline for comparison against binary-FNO/PSO/DE.

Unlike PSO/DE/FNO, GA is naturally a binary/combinatorial search (Holland
1975): chromosomes are directly boolean feature masks, no continuous
search space + transfer function is needed. This module therefore does NOT
go through binary_wrapper.py -- it implements the binary search directly --
but exposes the SAME result interface (best_mask, best_fitness, n_selected,
convergence_curve, n_evaluations) as binary_wrapper.BinaryWrapperResult so
batch_runner.py can treat all four algorithms uniformly.

Design (standard, textbook GA -- not tuned for this project):
  - Tournament selection (size 2)
  - Uniform crossover, probability pc = 0.8
  - Bit-flip mutation, probability pm = 1/n_features (standard rule of thumb)
  - Elitism: the single best individual is always carried over unchanged
  - Population re-evaluated each generation; budget = max_fes total
    fitness evaluations (same accounting convention as the other algorithms
    in this package, so comparisons are on equal footing)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class BinaryGAResult:
    best_mask: np.ndarray
    best_fitness: float
    n_selected: int
    convergence_curve: np.ndarray
    n_evaluations: int
    continuous_result: None = None  # interface parity with BinaryWrapperResult


class BinaryGA:
    def __init__(
        self,
        fitness_func: Callable[[np.ndarray], float],
        n_features: int,
        n_pop: int = 20,
        max_fes: int = 5000,
        seed: Optional[int] = None,
        pc: float = 0.8,
        pm: Optional[float] = None,  # default: 1/n_features
        tournament_size: int = 2,
        min_selected: int = 1,
    ):
        self.fitness_func = fitness_func
        self.n_features = n_features
        self.n_pop = n_pop
        self.max_fes = max_fes
        self.rng = np.random.default_rng(seed)
        self.pc = pc
        self.pm = pm if pm is not None else 1.0 / n_features
        self.tournament_size = tournament_size
        self.min_selected = min_selected

    def _repair(self, mask: np.ndarray) -> np.ndarray:
        if mask.sum() < self.min_selected:
            zero_idx = np.where(~mask)[0]
            need = self.min_selected - int(mask.sum())
            pick = self.rng.choice(zero_idx, size=min(need, len(zero_idx)), replace=False)
            mask[pick] = True
        return mask

    def _init_population(self):
        pop = self.rng.random((self.n_pop, self.n_features)) < 0.5
        pop = np.array([self._repair(ind.copy()) for ind in pop])
        fitness = np.array([self.fitness_func(ind) for ind in pop])
        return pop, fitness

    def _tournament_select(self, pop, fitness) -> np.ndarray:
        idx = self.rng.choice(self.n_pop, size=self.tournament_size, replace=False)
        winner = idx[np.argmin(fitness[idx])]  # minimization
        return pop[winner]

    def _crossover(self, parent1, parent2):
        if self.rng.random() < self.pc:
            mask = self.rng.random(self.n_features) < 0.5  # uniform crossover
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        return child1, child2

    def _mutate(self, ind: np.ndarray) -> np.ndarray:
        flip = self.rng.random(self.n_features) < self.pm
        ind = ind.copy()
        ind[flip] = ~ind[flip]
        return ind

    def optimize(self) -> BinaryGAResult:
        pop, fitness = self._init_population()
        fes = self.n_pop

        best_idx = np.argmin(fitness)
        best_mask = pop[best_idx].copy()
        best_fit = fitness[best_idx]
        curve = [best_fit] * fes

        while fes < self.max_fes:
            new_pop = [best_mask.copy()]  # elitism

            while len(new_pop) < self.n_pop and fes < self.max_fes:
                p1 = self._tournament_select(pop, fitness)
                p2 = self._tournament_select(pop, fitness)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._repair(self._mutate(c1))
                new_pop.append(c1)
                if len(new_pop) < self.n_pop and fes + 1 <= self.max_fes:
                    c2 = self._repair(self._mutate(c2))
                    new_pop.append(c2)

            new_pop = np.array(new_pop[: self.n_pop])
            new_fitness = np.empty(len(new_pop))
            new_fitness[0] = best_fit  # elite individual, already evaluated
            for i in range(1, len(new_pop)):
                if fes >= self.max_fes:
                    new_fitness[i] = np.inf  # unevaluated filler, won't be selected as best
                    continue
                new_fitness[i] = self.fitness_func(new_pop[i])
                fes += 1
                if new_fitness[i] < best_fit:
                    best_fit = new_fitness[i]
                    best_mask = new_pop[i].copy()
                curve.append(best_fit)

            pop, fitness = new_pop, new_fitness

        return BinaryGAResult(
            best_mask=best_mask,
            best_fitness=best_fit,
            n_selected=int(best_mask.sum()),
            convergence_curve=np.array(curve[: self.max_fes]),
            n_evaluations=fes,
        )