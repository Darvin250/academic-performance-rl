import numpy as np
import random

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

def step(time, energy, action):
    
    # energy change
    if action == 2:  # sleep
        energy = min(2, energy + 1)
    else:
        energy = max(0, energy - 1)

    # reward
    if time == 1:  # next is exam
        if energy == 2:
            reward = 50
        elif energy == 1:
            reward = 30
        else:
            reward = -20
    else:
        if energy == 0 and action != 2:
            reward = -5
        else:
            if action == 0:
                reward = 10
            elif action == 1:
                reward = 8
            else:
                reward = 5

    next_time = time - 1
    done = next_time == 0

    return next_time, energy, reward, done

for ep in range(episodes):

    time = 4
    energy = 2

    while time > 0:

        # choose action
        if random.uniform(0,1) < epsilon:
            action = random.randint(0,2)
        else:
            action = np.argmax(Q[time, energy])

        next_time, next_energy, reward, done = step(time, energy, action)

        # Q update
        old_q = Q[time, energy, action]
        future_q = np.max(Q[next_time, next_energy])

        Q[time, energy, action] = old_q + alpha * (reward + gamma * future_q - old_q)

        time, energy = next_time, next_energy

print("Learned Policy:")
for t in range(4, 0, -1):
    for e in range(3):
        best = np.argmax(Q[t, e])
        print(f"Time {t}, Energy {e} -> Action {best}")


print("\n--- Simulation of Learned Strategy ---")

time = 4
energy = 2
total_reward = 0

while time > 0:
    action = np.argmax(Q[time, energy])

    if action == 0:
        action_name = "Study"
    elif action == 1:
        action_name = "Past Year"
    else:
        action_name = "Sleep"

    print(f"Time: {time}, Energy: {energy}, Action: {action_name}")

    time, energy, reward, done = step(time, energy, action)
    total_reward += reward

print(f"Final Energy at Exam: {energy}")
print(f"Total Reward: {total_reward}")

