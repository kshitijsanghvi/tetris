#%% Initialize
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
    val_scores_record = []
    val_steps_record = []
    
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
                                    render_delay=None)
            
            current_state = next_states[best_action]
            steps += 1
        
        val_scores_record.append(env.get_game_score())
        val_steps_record.append(steps)
    
    mean_score = np.mean(val_scores_record)
    mean_steps = np.mean(val_steps_record)
    
    print()
    print("---------------")
    print("Validation")
    print()
    print("Max Score:\t" + str(max(val_scores_record)))
    print("Game Score:\t" + str(mean_score) )
    print("Steps:\t\t" + str(mean_steps) )
    print("---------------")
    print()
    
    return val_scores_record, val_steps_record


def dqn():        
    SAVE_MODEL_IF_SCORE_GREATER_THAN = 100

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
            val_scores.append(np.mean(val_score))
            val_steps.append(np.mean(val_step))
    
        wall_time.append(time())
        
        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])
            val_avg_score = mean(val_score)
            val_min_score = min(val_score)
            val_max_score = max(val_score)
            
            log.log(episode, avg_score=avg_score, min_score=min_score,
                    max_score=max_score, val_avg_score=val_avg_score,
                    val_min_score=val_min_score, val_max_score=val_max_score,
                    wall_time=time() - start_time)
            
            if val_max_score > SAVE_MODEL_IF_SCORE_GREATER_THAN:
                # save_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                agent.model.save(f'models/tetris-epsilon={epsilon}-epsilon_min={epsilon_min}-epsilon_stop_episode={epsilon_stop_episode}-episode={episode}-score-{val_max_score}-{datetime.now().strftime("%Y%m%d-%H%M%S")}.h5')
        
        

#%% Models
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
train_every = 1
                                 
validate_every = 25
num_val_reps = 5
validate_render = True
                                 
n_neurons = [32,32]
activations = ['relu', 'relu', 'linear']
add_batch_norm = True
use_target_model = False
update_target_every = None
                                 
agent = DQNAgent(env.get_state_size(),
                 n_neurons=n_neurons, activations=activations, add_batch_norm=add_batch_norm,
                 epsilon=epsilon, epsilon_min=epsilon_min,use_target_model=use_target_model,
                 update_target_every=update_target_every,
                 epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                 discount=discount, replay_start_size=replay_start_size)

log_dir = f'logs/tetris-epsilon={epsilon}-epsilon_min={epsilon_min}-epsilon_stop_episode={epsilon_stop_episode}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
log = CustomTensorBoard(log_dir=log_dir)

start_time = time()
wall_time = []
scores = []
val_scores = []
val_steps = []

dqn()


#%% Analyze

plt.plot(range(len(val_scores)), val_scores)
plt.plot(range(len(scores)), scores)


