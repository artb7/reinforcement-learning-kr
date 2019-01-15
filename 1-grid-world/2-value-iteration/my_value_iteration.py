# -*- coding: utf-8 -*-
from environment import GraphicDisplay, Env


class ValueIteration:
    def __init__(self, env):
        self.env = env
        self.value_table = [[0.0] * env.width
                            for _ in range(env.height)]
        self.discount_factor = 0.9

    def value_iteration(self):
        new_value_table = [[0.0] * self.env.width
                            for _ in range(self.env.height)]

        for state in self.env.get_all_states():

            if state == [2, 2]:
                continue

            max_value = -99999
            for action in self.env.possible_actions:
                reward = self.env.get_reward(state, action)
                next_state = self.env.state_after_action(state, action)
                next_value = self.get_value(next_state)
                value = reward + self.discount_factor * next_value
                if value > max_value:
                    max_value = value

            new_value_table[state[0]][state[1]] = round(max_value, 2)

        self.value_table = new_value_table

    def get_action(self, state):
        if state == [2, 2]:
            return []

        actions = []
        max_value = -99999
        for action in self.env.possible_actions:
            reward = self.env.get_reward(state, action)
            next_state = self.env.state_after_action(state, action)
            next_value = self.get_value(next_state)
            value = reward + self.discount_factor * next_value
            if value > max_value:
                max_value = value
                actions.clear()
                actions.append(action)
            elif value == max_value:
                actions.append(action)

        return actions

    def get_value(self, state):
        return self.value_table[state[0]][state[1]]


if __name__ == '__main__':
    env = Env()
    agent = ValueIteration(env)
    grid_world = GraphicDisplay(agent)
    grid_world.mainloop()
