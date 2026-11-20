import random
import argparse
import numpy as np
from ai_game import Game
from settings import *
import os

class Individual:
    """Individual in population of Genetic Algorithm.
    Attributes:
        genes: A list which can transform to weight of Neural Network.
        score: Score of the snake played by its Neural Network.
        steps: Steps of the snake played by its Neural Network.
        fitnees: Fitness of Individual.
        seed: The random seed of the game, saved for reproduction.
    """
    def __init__(self, genes):
        self.genes = genes
        self.score = 0
        self.steps = 0
        self.fitness = 0
        self.shaped = 0
        self.seed = None

    def get_fitness(self, reward_mode="shaped"):
        """Evaluate the individual; fitness formula chosen by reward_mode."""
        game = Game([self.genes])
        self.score, self.steps, self.shaped, self.seed = game.play()
        if reward_mode == "simple":
            self.fitness = (self.score + 1.0 / self.steps) * 100000
        else:
            self.fitness = self.shaped * 100000

class GA:
    """Genetic Algorithm.
    Attributes:
        p_size: Size of the parent generation.
        c_size: Size of the child generation.
        genes_len: Length of the genes.
        mutate_rate: Probability of the mutation.
        population: A list of individuals.
        best_individual: Individual with best fitness.
        avg_score: Average score of the population.
    """
    def __init__(self, p_size=P_SIZE, c_size=C_SIZE, genes_len=GENES_LEN, mutate_rate=MUTATE_RATE):
        self.p_size = p_size
        self.c_size = c_size
        self.genes_len = genes_len
        self.mutate_rate = mutate_rate
        self.population = []
        self.best_individual = None
        self.avg_score = 0
        self.generation = 0

        # Mutation / diversity knobs. Fixed mild mutation works best here:
        # adaptive decay was shown to stall in a low-score local optimum.
        self.adaptive = False
        self.base_rate, self.min_rate = 0.1, 0.03
        self.base_sigma, self.min_sigma = 0.2, 0.05
        self.fixed_sigma = 0.2
        self.immigrant_ratio = 0.0
        self.selection = "roulette"    # "tournament" or "roulette"
        self.reward_mode = "simple"    # "shaped" or "simple"

    def generate_ancestor(self):
        for i in range(self.p_size):
            genes = np.random.uniform(-1, 1, self.genes_len)
            self.population.append(Individual(genes))
    
    def inherit_ancestor(self):
        """Load genes from './genes/all/{i}', i: the ith individual."""
        for i in range(self.p_size):
            pth = os.path.join("genes", "all", str(i))
            with open(pth, "r") as f:
                genes = np.array(list(map(float, f.read().split())))
                self.population.append(Individual(genes))

    def crossover(self, c1_genes, c2_genes):
        """Single point crossover."""
        point = np.random.randint(0, self.genes_len)
        c1_genes[:point + 1], c2_genes[:point + 1] = c2_genes[:point + 1], c1_genes[:point + 1]

    def mutate(self, c_genes):
        """Gaussian mutation; rate and step size adapt to generation if enabled."""
        if self.adaptive:
            rate = max(self.min_rate, self.base_rate * 0.985 ** self.generation)
            sigma = max(self.min_sigma, self.base_sigma * 0.99 ** self.generation)
        else:
            rate, sigma = self.mutate_rate, self.fixed_sigma
        mask = np.random.random(c_genes.shape) < rate
        c_genes[mask] += np.random.normal(scale=sigma, size=int(mask.sum()))

    def elitism_selection(self, size):
        """Select the top #size individuals to be parents."""
        population = sorted(self.population, key =lambda individual: individual.fitness, reverse=True)
        return population[:size]

    def tournament_selection(self, size, k=3):
        """Tournament selection: steadier pressure, less prone to premature."""
        selection = []
        pool = min(k, len(self.population))
        for _ in range(size):
            contenders = random.sample(self.population, pool)
            selection.append(max(contenders, key=lambda individual: individual.fitness))
        return selection

    def roulette_wheel_selection(self, size):
        """Fitness-proportionate selection (fitness shifted to non-negative)."""
        fits = np.array([individual.fitness for individual in self.population],
                        dtype=float)
        fits = fits - fits.min()
        wheel = fits.sum()
        if wheel <= 0:
            return random.sample(self.population, size)
        probs = fits / wheel
        picks = np.random.choice(len(self.population), size=size,
                                 replace=True, p=probs)
        return [self.population[i] for i in picks]

    def evolve(self):
        '''The main procss of Genetic Algorithm.'''
        sum_score = 0
        for individual in self.population:
            individual.get_fitness(self.reward_mode)
            sum_score += individual.score
        self.avg_score = sum_score / len(self.population)

        self.population = self.elitism_selection(self.p_size)  # Select parents to generate children.
        self.best_individual = self.population[0]
        random.shuffle(self.population)

        # Generate children; a few random immigrants keep diversity high.
        n_immigrants = int(self.c_size * self.immigrant_ratio)
        children = []
        while len(children) < self.c_size:
            if n_immigrants > 0 and len(children) < n_immigrants:
                genes = np.random.uniform(-1, 1, self.genes_len)
                children.append(Individual(genes))
                continue
            selector = (self.tournament_selection if self.selection == "tournament"
                        else self.roulette_wheel_selection)
            p1, p2 = selector(2)
            c1_genes, c2_genes = p1.genes.copy(), p2.genes.copy()
            self.crossover(c1_genes, c2_genes)
            self.mutate(c1_genes)
            self.mutate(c2_genes)
            c1, c2 = Individual(c1_genes), Individual(c2_genes)
            children.extend([c1, c2])

        random.shuffle(children)
        self.population.extend(children)
        self.generation += 1

    def save_best(self):
        """Save the best individual that can get #score score so far."""
        score = self.best_individual.score
        genes_pth= os.path.join("genes", "best", str(score))
        with open(genes_pth, "w") as f:
            for gene in self.best_individual.genes:
                f.write(str(gene) + " ") 
        seed_pth = os.path.join("seed", str(score))
        with open(seed_pth, "w") as f:
            f.write(str(self.best_individual.seed)) 

    def save_all(self):
        """Save the population (elite parents, sorted by fitness)."""
        for individual in self.population:
            individual.get_fitness()
        population = self.elitism_selection(self.p_size)
        for i in range(len(population)):
            pth = os.path.join("genes", "all", str(i))
            with open(pth, "w") as f:
                for gene in population[i].genes:
                    f.write(str(gene) + " ")

    def save_checkpoint(self, directory):
        """Save elite parents and generation so training can resume later."""
        os.makedirs(directory, exist_ok=True)
        pop_dir = os.path.join(directory, "pop")
        os.makedirs(pop_dir, exist_ok=True)
        elites = self.elitism_selection(self.p_size)
        for i, individual in enumerate(elites):
            with open(os.path.join(pop_dir, str(i)), "w") as f:
                f.write(" ".join(map(str, individual.genes)))
        with open(os.path.join(directory, "meta.txt"), "w") as f:
            f.write(f"{self.generation} {self.best_individual.score} "
                    f"{self.best_individual.seed}")

    def load_checkpoint(self, directory):
        """Restore population and generation from a checkpoint.

        Returns the saved (record, best_seed), or None if no checkpoint.
        """
        pop_dir = os.path.join(directory, "pop")
        meta_path = os.path.join(directory, "meta.txt")
        if not (os.path.isdir(pop_dir) and os.path.exists(meta_path)):
            return None
        # Guard against checkpoints produced by a different network size.
        with open(os.path.join(pop_dir, "0"), "r") as probe:
            if len(probe.read().split()) != self.genes_len:
                return None
        self.population = []
        for i in range(self.p_size):
            with open(os.path.join(pop_dir, str(i)), "r") as f:
                genes = np.array(list(map(float, f.read().split())))
            self.population.append(Individual(genes))
        with open(meta_path, "r") as f:
            generation, record, best_seed = map(int, f.read().split())
        self.generation = generation
        self.best_individual = self.population[0]
        return record, best_seed

    def seed_from_genes(self, base_genes, diversity=0.2):
        """Build a population around one strong genome (warm start)."""
        base = np.array(base_genes)
        self.population = []
        for i in range(self.p_size):
            genes = base.copy()
            if i > 0:
                mask = np.random.random(self.genes_len) < 0.3
                genes[mask] += np.random.normal(scale=diversity, size=int(mask.sum()))
            self.population.append(Individual(genes))
        self.best_individual = self.population[0]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--inherit', action="store_true", 
                         help="Whether to load genes from path ./genes/all.")

    parser.add_argument('-s', '--show', action="store_true", help='whether to show the best individual to play snake after each envolve.')
    args = parser.parse_args()

    ga = GA()
    
    if args.inherit:
        ga.inherit_ancestor()
    else:
        ga.generate_ancestor()

    generation = 0
    record = 0
    while True:
        generation += 1
        ga.evolve()
        print("generation:", generation, ", record:", record,
              ", best score:", ga.best_individual.score, ", average score:", ga.avg_score)
        if ga.best_individual.score >= record:
            record = ga.best_individual.score 
            ga.save_best()
        if args.show:
            genes = ga.best_individual.genes
            seed = ga.best_individual.seed
            game = Game(show=True, genes_list=[genes], seed=seed)
            game.play()
    
        # Save the population every 20 generation.
        if generation % 20 == 0:
            ga.save_all()
