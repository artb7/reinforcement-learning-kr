# -*- coding: utf-8 -*-
import random
from environment import GraphicDisplay, Env

class PolicyIteration:
    def __init__(self, env):

        # Initialize environment
        self.env = env

        # Initialize value
        self.value_table = [[0.0] * env.width for _ in range(env.height)]

        # Initialize policy
        self.policy_table = ([[[0.25, 0.25, 0.25, 0.25]] * env.width 
                             for _ in range(env.height)])
        self.policy_table[2][2] = []

        # Initialize discount factor 
        self.discount_factor = 0.9

    def policy_evaluation(self):
        new_value_table = [[0.0] * self.env.width for _ in range(self.env.height)]

        for state in self.env.get_all_states():

            if state == [2, 2]:
                continue

            value = 0.0
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                value_of_next_state = self.get_value(next_state)
                policy = self.get_policy(state)[action]
                 
                value += policy * (reward + self.discount_factor * value_of_next_state)
            
            new_value_table[state[0]][state[1]] = round(value, 2)

        self.value_table = new_value_table

    def policy_improvement(self):
#        new_policy = ([[[0.0, 0.0, 0.0, 0.0]] * self.env.width 
#                      for _ in range(self.env.height)])
        #next_policy = self.policy_table
    
        for state in self.env.get_all_states():

            if state == [2, 2]:
                continue

            max_index = []
            max_value = -999999
            result = [0.0, 0.0, 0.0, 0.0]

            for index, action in enumerate(self.env.possible_actions):
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                value_of_next_state = self.get_value(next_state)
                value = reward + self.discount_factor * value_of_next_state
                if max_value < value:
                    max_value = value
                    max_index.clear()
                    max_index.append(index)
                elif max_value == value:
                    max_index.append(index)

            prob = 1. / len(max_index)

            for index in max_index:
                result[index] = prob

            self.policy_table[state[0]][state[1]] = result
                
        #self.policy_table = next_policy
    
    def get_action(self, state):
        random_pick = random.randrange(100) / 100
        policy = self.get_policy(state)
        policy_sum = 0.0
        for index, value in enumerate(policy):
            policy_sum += value
            if random_pick < policy_sum:
                return index

    def get_policy(self, state):
        return self.policy_table[state[0]][state[1]]

    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)



if __name__ == '__main__':
    env = Env()
    agent = PolicyIteration(env)
    grid_world = GraphicDisplay(agent)
    grid_world.mainloop()

