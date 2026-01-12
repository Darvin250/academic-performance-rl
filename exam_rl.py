import numpy as np
import random

class ExamEnvironment:
    """
    Environment for exam preparation under time and energy constraints.
    State: (time_left, energy_level)
    Actions:
        0 = Study
        1 = Past Year Questions
        2 = Sleep
    """

    def __init__(self):
        self.max_time = 4
        self.max_energy = 2
        self.reset()

    def reset(self):
        self.time = self.max_time
        self.energy = self.max_energy
        return (self.time, self.energy)

    def step(self, action):
        # Energy transition
        if action == 2:  # Sleep
            self.energy = min(self.max_energy, self.energy + 1)
        else:  # Study or Past Year
            self.energy = max(0, self.energy - 1)

        # Reward calculation
        if self.time == 1:  # Exam is next
            if self.energy == 2:
                reward = 50
            elif self.energy == 1:
                reward = 30
            else:
                reward = -20
        else:
            if self.energy == 0 and action != 2:
                reward = -5
            else:
                reward = [10, 8, 5][action]

        # Time transition
        self.time -= 1
        done = self.time == 0

        return (self.time, self.energy), reward, done


# Time and energy levels
TIME_LEVELS = 5   # 0 to 4
ENERGY_LEVELS = 3 # 0 to 2
ACTIONS = 3       # study, past year, sleep

# Q-table: [time][energy][action]
Q = np.zeros((TIME_LEVELS, ENERGY_LEVELS, ACTIONS))

# Hyperparameters
alpha = 0.1       # learning rate
gamma = 0.9       # discount factor
epsilon = 0.2     # exploration rate
episodes = 1000

env = ExamEnvironment()

for ep in range(episodes):
    state = env.reset()

    while True:
        time, energy = state

        # Action selection (epsilon-greedy)
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, ACTIONS - 1)
        else:
            action = np.argmax(Q[time, energy])

        next_state, reward, done = env.step(action)
        next_time, next_energy = next_state

        # Q-learning update
        old_q = Q[time, energy, action]
        future_q = np.max(Q[next_time, next_energy])
        Q[time, energy, action] = old_q + alpha * (reward + gamma * future_q - old_q)

        state = next_state

        if done:
            break

def time_label(t):
    return {4: "Far (plenty)", 3: "Moderate", 2: "Close", 1: "Exam next"}.get(t, "Unknown")


def energy_label(e):
    return {0: "Low", 1: "Medium", 2: "High"}.get(e, "Unknown")

action_names = ["Study", "Past Year", "Sleep"]

# Learned policy presented as a table
print("Learned Policy:")
header = f"{'Time':<6}{'Time Meaning':<22}{'Energy':<8}{'Energy Meaning':<15}{'Action':<8}{'Action Meaning'}"
print(header)
print("-" * len(header))
for t in range(4, 0, -1):
    for e in range(3):
        best_action = int(np.argmax(Q[t, e]))
        action_name = action_names[best_action]
        print(f"{t:<6}{time_label(t):<22}{e:<8}{energy_label(e):<15}{best_action:<8}{action_name}")


# Simulation presented as a table
print("\n--- Simulation of Learned Strategy ---")
print(header)
print("-" * len(header))

state = env.reset()
total_reward = 0

while True:
    time, energy = state
    action = int(np.argmax(Q[time, energy]))
    action_name = action_names[action]
    print(f"{time:<6}{time_label(time):<22}{energy:<8}{energy_label(energy):<15}{action:<8}{action_name}")

    state, reward, done = env.step(action)
    total_reward += reward

    if done:
        break

final_energy = state[1]
print("\nSummary:")
print(f"Final Energy at Exam: {final_energy} ({energy_label(final_energy)})")
print(f"Total Reward: {total_reward}")