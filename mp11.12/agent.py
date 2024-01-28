import numpy as np
import utils
import copy
# import random


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def state_indices(self, state):
        FOOD_DIR_X, FOOD_DIR_Y, ADJOINING_WALL_X_STATES, ADJOINING_WALL_Y_STATES, ADJOINING_BODY_TOP_STATES, ADJOINING_BODY_BOTTOM_STATES, ADJOINING_BODY_LEFT_STATES,ADJOINING_BODY_RIGHT_STATES, ACTIONS = state
        return (FOOD_DIR_X, FOOD_DIR_Y, ADJOINING_WALL_X_STATES, ADJOINING_WALL_Y_STATES, ADJOINING_BODY_TOP_STATES, ADJOINING_BODY_BOTTOM_STATES, ADJOINING_BODY_LEFT_STATES,ADJOINING_BODY_RIGHT_STATES, ACTIONS)

    def update_n(self, state, action):
        # TODO - MP11: Update the N-table. 
        # _state_indices = self.state_indices(state)
        # self.N[_state_indices][action] += 1
        if state == None or action == None:
            return 
        state_plus_action = state + (action,)
        self.N[state_plus_action] += 1

    def update_q(self, s, a, r, s_prime):
        # TODO - MP11: Update the Q-table. 
        if s == None or a == None or s_prime == None:
            return
        alpha = self.C/(self.C + self.N[s + (a,)])
        max_Q = -9999999
        for action in self.actions:
            max_Q = max(max_Q, self.Q[s_prime + (action,)])
        self.Q[s + (a,)] += alpha*(r + self.gamma*max_Q - self.Q[s + (a,)])
           

    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        if dead:
            self.reset()
            return utils.UP
        
        f_function = []
        direction = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        snake_head = environment[0], environment[1]
        food = environment[3], environment[4]
        rock = environment[5], environment[6]
        next_points = {}
        s_prime = {}
        for action in self.actions:
            next_env = copy.deepcopy(environment)
            next_head = (snake_head[0] + direction[action][0], snake_head[1] + direction[action][1])
            next_points[action] = points
            next_env[2].append(snake_head)
            if next_head == food:
                next_points[action] += 1
            elif next_head == rock:
                del next_env[2][0]
                next_points[action] -= 1
            else:
                del next_env[2][0]
                next_points[action] -= 0.1
            next_env[0], environment[1] = next_head
            s_prime[action] = self.generate_state(next_env)
            N_value = self.N[s_prime[action] + (action,)]
            Q_value = self.Q[s_prime[action] + (action,)]
            if N_value < self.Ne:
                f_function.append(1)
            else:
                f_function.append(Q_value)
        max_f = max(f_function)
        best_actions = [action for action, value in enumerate(f_function) if value == max_f]
        order = {utils.UP: 0, utils.DOWN: 1, utils.LEFT: 2, utils.RIGHT: 3}
        _act = max(best_actions, key=lambda x: (f_function[x], order[x]))
        if self._train:
            self.update_n(self.s, _act)
            self.update_q(self.s, _act, next_points[_act], s_prime[_act])
        self.s = s_prime[_act]
        self.a = _act
        return _act
        



        # TODO - MP12: write your function here

        # return random.randint(0, 3)

    def generate_state(self, environment):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        '''
        # state:FOOD_DIR_X, FOOD_DIR_Y, ADJOINING_WALL_X_STATES, ADJOINING_WALL_Y_STATES, ADJOINING_BODY_TOP_STATES, ADJOINING_BODY_BOTTOM_STATES, ADJOINING_BODY_LEFT_STATES,ADJOINING_BODY_RIGHT_STATES
        # TODO - MP11: Implement this helper function that generates a state given an environment 
        snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y = environment
        # food_x
        if food_x < snake_head_x:
            FOOD_DIR_X = 1
        elif food_x > snake_head_x:
            FOOD_DIR_X = 2
        else:
            FOOD_DIR_X = 0
        # food_y
        if food_y < snake_head_y:
            FOOD_DIR_Y = 1
        elif food_y > snake_head_y:
            FOOD_DIR_Y = 2
        else:
            FOOD_DIR_Y = 0
        # wall_head_x
        if snake_head_x <= 1 or ((rock_x == snake_head_x - 1 or rock_x + 1 == snake_head_x - 1) and (rock_y == snake_head_y)):
            ADJOINING_WALL_X_STATES = 1
        elif snake_head_x >= self.display_width - 2 or ((rock_x == snake_head_x + 1 or rock_x + 1 == snake_head_x + 1) and (rock_y == snake_head_y)):
            ADJOINING_WALL_X_STATES = 2
        else:
            ADJOINING_WALL_X_STATES = 0
        # wall_head_y
        if snake_head_y <= 1 or (rock_y == snake_head_y - 1 and (rock_x == snake_head_x or rock_x + 1 == snake_head_x)):
            ADJOINING_WALL_Y_STATES = 1
        elif snake_head_y >= self.display_height - 2 or (rock_y == snake_head_y + 1 and (rock_x == snake_head_x or rock_x + 1 == snake_head_x)):
            ADJOINING_WALL_Y_STATES = 2
        else:
            ADJOINING_WALL_Y_STATES = 0
        # snake_body
        ADJOINING_BODY_TOP_STATES, ADJOINING_BODY_BOTTOM_STATES, ADJOINING_BODY_LEFT_STATES,ADJOINING_BODY_RIGHT_STATES = 0, 0, 0, 0
        for ele in snake_body:
            ele_x, ele_y = ele
            if snake_head_x + 1 == ele_x and snake_head_y == ele_y:
                ADJOINING_BODY_RIGHT_STATES = 1
            if snake_head_x - 1 == ele_x and snake_head_y == ele_y:
                ADJOINING_BODY_LEFT_STATES = 1
            if snake_head_x == ele_x and snake_head_y - 1 == ele_y:
                ADJOINING_BODY_TOP_STATES = 1
            if snake_head_x == ele_x and snake_head_y + 1 == ele_y:
                ADJOINING_BODY_BOTTOM_STATES = 1
        return (FOOD_DIR_X, FOOD_DIR_Y, ADJOINING_WALL_X_STATES, ADJOINING_WALL_Y_STATES, ADJOINING_BODY_TOP_STATES, ADJOINING_BODY_BOTTOM_STATES, ADJOINING_BODY_LEFT_STATES,ADJOINING_BODY_RIGHT_STATES)
