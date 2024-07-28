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
pygame.display.set_caption("Neural Network Doodle Jump")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Adjustable parameters
POPULATION_SIZE = 100
PLATFORM_DISTANCE = 50

# Game objects
class Platform:
    def __init__(self, x, y, width=60, height=10, is_fixed=False):
        self.rect = pygame.Rect(x, y, width, height)
        self.is_fixed = is_fixed

    def draw(self, offset):
        color = BLUE if self.is_fixed else GREEN
        draw_rect = self.rect.copy()
        draw_rect.y -= offset
        pygame.draw.rect(screen, color, draw_rect)

class Player:
    def __init__(self):
        self.rect = pygame.Rect(WIDTH // 2, HEIGHT - 50, 30, 30)
        self.velocity = 0
        self.jump_power = -10
        self.gravity = 0.5
        self.score = 0

    def update(self, platforms, action):
        self.velocity += self.gravity
        self.rect.y += self.velocity

        if action == 1:
            self.rect.x -= 5
        elif action == 2:
            self.rect.x += 5

        for platform in platforms:
            if self.rect.colliderect(platform.rect) and self.velocity > 0:
                self.rect.bottom = platform.rect.top
                self.velocity = self.jump_power

        if self.rect.left < 0:
            self.rect.left = 0
        elif self.rect.right > WIDTH:
            self.rect.right = WIDTH

        self.score = max(self.score, HEIGHT - self.rect.y)

    def draw(self, offset):
        draw_rect = self.rect.copy()
        draw_rect.y -= offset
        pygame.draw.rect(screen, RED, draw_rect)

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
        self.population = [NeuralNetwork(4, [8, 8], 3) for _ in range(population_size)]

    def select_parents(self, fitness_scores):
        total_fitness = sum(fitness_scores)
        selection_probs = [f / total_fitness for f in fitness_scores]
        return random.choices(self.population, weights=selection_probs, k=2)

    def crossover(self, parent1, parent2):
        child = NeuralNetwork(4, [8, 8], 3)
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
    platforms = [
        Platform(WIDTH // 2 - 30, HEIGHT - 50, is_fixed=True),
        Platform(WIDTH // 4 - 30, HEIGHT - 100, is_fixed=True),  # Moved up
        Platform(3 * WIDTH // 4 - 30, HEIGHT - 150, is_fixed=True)  # Moved up
    ]
    for y in range(HEIGHT - 200, -50, -PLATFORM_DISTANCE):  # Adjusted distance between platforms
        platforms.append(Platform(random.randint(0, WIDTH - 60), y))
    return players, platforms

players, platforms = reset_game()
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
            players, platforms = reset_game()
            start_time = time.time()
            best_score = 0

    screen.fill(WHITE)

    # Determine the best player to follow
    best_player = max(players, key=lambda p: p.score)
    camera_offset = best_player.rect.y - HEIGHT // 2

    # Draw platforms
    for platform in platforms:
        platform.draw(camera_offset)

    # Update and draw players
    active_players = []
    for i, player in enumerate(players):
        if player.rect.top > HEIGHT:
            continue

        active_players.append(player)

        # Get input for neural network
        nearest_platform = min(platforms, key=lambda p: abs(p.rect.y - player.rect.y))
        inputs = [
            player.rect.x / WIDTH,
            player.rect.y / HEIGHT,
            nearest_platform.rect.x / WIDTH,
            nearest_platform.rect.y / HEIGHT
        ]

        # Get action from neural network
        action = ga.population[i].forward(inputs)

        # Update player
        player.update(platforms, action)
        player.draw(camera_offset)

        best_score = max(best_score, player.score)
        overall_best_score = max(overall_best_score, best_score)

    # Check if all players have fallen
    if not active_players:
        # Evolve the population
        fitness_scores = [player.score for player in players]
        ga.evolve(fitness_scores)
        generation += 1
        print(f"Generation: {generation}, Best Score: {max(fitness_scores)}")

        # Reset the game
        players, platforms = reset_game()

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

    layer_sizes = [4, 8, 8, 3]
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
