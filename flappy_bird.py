import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
import time

# Initialize Pygame
pygame.init()

# Set up the game window
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Network Flappy Bird")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Adjustable parameters
POPULATION_SIZE = 500
PIPE_GAP = 200
PIPE_WIDTH = 60
PIPE_DISTANCE = 200

# Game objects
class Pipe:
    def __init__(self, x, height, gap=PIPE_GAP):
        self.rect_top = pygame.Rect(x, 0, PIPE_WIDTH, height)
        self.rect_bottom = pygame.Rect(x, height + gap, PIPE_WIDTH, HEIGHT - height - gap)
        self.passed = False

    def update(self):
        self.rect_top.x -= 5
        self.rect_bottom.x -= 5

    def draw(self):
        pygame.draw.rect(screen, GREEN, self.rect_top)
        pygame.draw.rect(screen, GREEN, self.rect_bottom)

class Player:
    def __init__(self):
        self.rect = pygame.Rect(50, HEIGHT // 2, 30, 30)
        self.velocity = 0
        self.jump_power = -10
        self.gravity = 0.5
        self.score = 0

    def update(self, pipes, action):
        self.velocity += self.gravity
        if action == 1:
            self.velocity = self.jump_power
        self.rect.y += self.velocity

        if self.rect.top < 0:
            self.rect.top = 0
        elif self.rect.bottom > HEIGHT:
            self.rect.bottom = HEIGHT

        self.score += 1

        for pipe in pipes:
            if self.rect.colliderect(pipe.rect_top) or self.rect.colliderect(pipe.rect_bottom):
                return False

        return True

    def draw(self):
        pygame.draw.rect(screen, RED, self.rect)

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layers = [input_size] + hidden_sizes + [output_size]
        self.weights = [np.random.randn(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)]

    def forward(self, X):
        for weight in self.weights[:-1]:
            X = np.maximum(0, np.dot(X, weight))  # ReLU activation
        return np.argmax(np.dot(X, self.weights[-1]))

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, elite_size):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.population = [NeuralNetwork(4, [8, 8], 2) for _ in range(population_size)]

    def select_parents(self, fitness_scores):
        total_fitness = sum(fitness_scores)
        selection_probs = [f / total_fitness for f in fitness_scores]
        return random.choices(self.population, weights=selection_probs, k=2)

    def crossover(self, parent1, parent2):
        child = NeuralNetwork(4, [8, 8], 2)
        for i in range(len(child.weights)):
            mask = np.random.rand(*child.weights[i].shape) < 0.5
            child.weights[i] = np.where(mask, parent1.weights[i], parent2.weights[i])
        return child

    def mutate(self, network):
        for weight in network.weights:
            mask = np.random.rand(*weight.shape) < self.mutation_rate
            weight[mask] += np.random.randn(*weight.shape)[mask] * 0.1

    def evolve(self, fitness_scores):
        sorted_population = [x for _, x in sorted(zip(fitness_scores, self.population), key=lambda pair: pair[0], reverse=True)]
        new_population = sorted_population[:self.elite_size]

        while len(new_population) < self.population_size:
            parents = self.select_parents(fitness_scores)
            child = self.crossover(parents[0], parents[1])
            self.mutate(child)
            new_population.append(child)

        self.population = new_population

# Game setup
mutation_rate = 0.1
elite_size = 5
ga = GeneticAlgorithm(POPULATION_SIZE, mutation_rate, elite_size)
generation = 0

def reset_game():
    players = [Player() for _ in range(POPULATION_SIZE)]
    pipes = [Pipe(WIDTH, random.randint(100, HEIGHT - 100 - PIPE_GAP))]
    return players, pipes

players, pipes = reset_game()
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

# Timer setup
RESET_EVENT = pygame.USEREVENT + 1
pygame.time.set_timer(RESET_EVENT, 60000)  # 60000 milliseconds = 1 minute

start_time = time.time()
best_score = 0
overall_best_score = 0

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == RESET_EVENT:
            print(f"One minute passed. Resetting generation.")
            generation = 0
            players, pipes = reset_game()
            start_time = time.time()
            best_score = 0

    screen.fill(WHITE)

    if len(pipes) == 0 or pipes[-1].rect_top.right < WIDTH - PIPE_DISTANCE:
        pipes.append(Pipe(WIDTH, random.randint(100, HEIGHT - 100 - PIPE_GAP)))

    for pipe in pipes:
        pipe.update()
        pipe.draw()

    pipes = [pipe for pipe in pipes if pipe.rect_top.right > 0]

    active_players = []
    for i, player in enumerate(players):
        if not player.update(pipes, ga.population[i].forward([player.rect.y / HEIGHT, player.velocity, (pipes[0].rect_top.x - player.rect.x) / WIDTH, (pipes[0].rect_top.bottom - player.rect.y) / HEIGHT])):
            continue

        active_players.append(player)
        player.draw()

        best_score = max(best_score, player.score)
        overall_best_score = max(overall_best_score, best_score)

    if not active_players:
        # Evolve the population
        fitness_scores = [player.score for player in players]
        ga.evolve(fitness_scores)
        generation += 1
        print(f"Generation: {generation}, Best Score: {max(fitness_scores)}")

        # Reset the game
        players, pipes = reset_game()

    # Display generation, score, and time
    gen_text = font.render(f"Generation: {generation}", True, BLACK)
    score_text = font.render(f"Best Score: {int(best_score)}", True, BLACK)
    overall_score_text = font.render(f"Overall Best: {int(overall_best_score)}", True, BLACK)
    time_left = 60 - int(time.time() - start_time)
    time_text = font.render(f"Time Left: {time_left}s", True, BLACK)
    screen.blit(gen_text, (10, 10))
    screen.blit(score_text, (10, 50))
    screen.blit(overall_score_text, (10, 90))
    screen.blit(time_text, (10, 130))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

# Create neural network diagram with weights
def create_nn_diagram():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("Neural Network Structure with Weights")

    layer_sizes = [4, 8, 8, 2]
    layer_positions = [0, 1, 2, 3]
    
    for i, layer_size in enumerate(layer_sizes):
        for j in range(layer_size):
            circle = plt.Circle((i, j * 0.3), 0.1, fill=False)
            ax.add_artist(circle)
            
            if i < len(layer_sizes) - 1:
                weights = ga.population[0].weights[i]
                for k in range(layer_sizes[i + 1]):
                    weight = weights[j, k]
                    color = 'red' if weight < 0 else 'green'
                    ax.plot([i, i + 1], [j * 0.3, k * 0.3], color, linewidth=abs(weight) * 2)
                    ax.text((i + i + 1) / 2, (j * 0.3 + k * 0.3) / 2, f"{weight:.2f}", 
                            ha='center', va='center', fontsize=8)

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 2.5)
    ax.axis('off')
    
    plt.text(0, -0.7, "Input\nLayer", ha='center')
    plt.text(1, -0.7, "Hidden\nLayer 1", ha='center')
    plt.text(2, -0.7, "Hidden\nLayer 2", ha='center')
    plt.text(3, -0.7, "Output\nLayer", ha='center')

    plt.tight_layout()
    plt.show()

create_nn_diagram()
