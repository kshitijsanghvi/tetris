#%% Learn

from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean, median
import random
from logs import CustomTensorBoard
from tqdm import tqdm
import cv2
from time import time
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



def experience(render=False, max_steps=200):
    current_state = env.reset()
    done = False
    steps = 0
    
    while not done and (not max_steps or steps < max_steps):
        next_states = env.get_next_states()
        best_state = agent.best_state(next_states.values(), exploration=True)
        
        best_action = None
        for action, state in next_states.items():
            if state == best_state:
                best_action = action
                break

        reward, done = env.play(best_action[0], best_action[1], render=render,
                                render_delay=None)
        
        agent.add_to_memory(current_state, next_states[best_action], reward, done)
        current_state = next_states[best_action]
        steps += 1
    


def validate(num_reps, validate_render=True, max_steps=200):
    temp_scores = []
    temp_steps = []
    
    for rep in range(num_reps):
        if validate_render and not rep:
            render = True
        else:
            render = False
            
        current_state = env.reset()
        done = False
        steps = 0
        
        while not done and (not max_steps or steps < max_steps):
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values(), exploration=False)
            
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break
    
            reward, done = env.play(best_action[0], best_action[1], render=render,
                                    render_delay=render_delay)
            
            current_state = next_states[best_action]
            steps += 1
        
        temp_scores.append(env.get_game_score())
        temp_steps.append(steps)
    
    mean_score = np.mean(temp_scores)
    mean_steps = np.mean(temp_steps)
    
    print()
    print("---------------")
    print("Validation")
    print()
    print("Game Score:\t" + str(mean_score) )
    print("Steps:\t\t" + str(mean_steps) )
    print("---------------")
    print()
    
    return mean_score, mean_steps




def dqn():
    for episode in tqdm(range(1,episodes+1)):

        # Game
        experience()
        scores.append(env.get_game_score())
        
        # Train
        if episode % train_every == 0:
            agent.train(memory_batch_size=memory_batch_size, training_batch_size=training_batch_size, epochs=epochs)
    
    
        # Validate
        if episode % validate_every == 0:
            val_score, val_step = validate(num_val_reps,validate_render,max_steps )
            val_scores.append(val_score)
            val_steps.append(val_step)
    
        wall_time.append(time())
        
        


env = Tetris()
episodes = 2000
max_steps = 200
epsilon = 1
epsilon_min = 0.1
epsilon_stop_episode = 1500
mem_size = 20000
discount = 0.95
training_batch_size = 512
memory_batch_size = 512
epochs = 1
replay_start_size = 2048

# render_every = None
# log_every = 50

train_every = 1

# render_delay = None

validate_every = 25
num_val_reps = 5
validate_render = True

n_neurons = [64,64]
activations = ['relu', 'relu', 'linear']
add_batch_norm = True

agent = DQNAgent(env.get_state_size(),
                 n_neurons=n_neurons, activations=activations, add_batch_norm=add_batch_norm,
                 epsilon=epsilon, epsilon_min=epsilon_min,
                 epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                 discount=discount, replay_start_size=replay_start_size)

start_time = time()
wall_time = []
scores = []
val_scores = []
val_steps = []

dqn()


    

#%% Save
save_time = datetime.now().strftime("%Y%m%d-%H%M%S")

results = { 'wall_time' : wall_time,
            'scores' : scores,
            'val_scores' : val_scores,
            'val_steps' : val_steps }

model_description = 'basic'

with open(f'./models/tetris-nn-{model_description}-{save_time}.pkl', 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    
agent.model.save(f'./models/tetris-nn-{model_description}-{save_time}.h5')

with open(f'./models/tetris-nn-{model_description}-{save_time}.txt', 'w') as f:
    print('', file=f)


# load_file = './models/tetris-nn-{model_description}-{save_time}'
# with open('./models/' + load_file + '.pkl, 'rb') as f:
#     results = pickle.load(f)
# model = tf.keras.models.load_model('./models/' + load_file + '.h5')


#%% Analyze

plt.plot(range(len(val_scores)), val_scores)
plt.plot(range(len(scores)), scores)


