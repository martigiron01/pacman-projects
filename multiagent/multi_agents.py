# multi_agents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattan_distance
from game import Directions, Actions
from pacman import GhostRules
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position()
        new_food = successor_game_state.get_food()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [ghostState.scared_timer for ghostState in new_ghost_states]
        
        "*** YOUR CODE HERE ***"

        # Initial score from successor state
        score = successor_game_state.get_score()
        
        # We calculate the distances to all food
        food_distances = [manhattan_distance(new_pos, food) for food in new_food.as_list()]

        # Then we check the nearest one
        if food_distances:
            nearest_food_distance = min(food_distances)
            # We increase score as closer to the nearest food
            score += 1.0 / nearest_food_distance

        # We iterate over all ghosts
        for i in range(len(new_ghost_states)):
            ghost = new_ghost_states[i]
            scared_time = new_scared_times[i]

            # We calculate distance to ghost
            ghost_distance = manhattan_distance(new_pos, ghost.get_position())
            
            # We decrease score as closer to the not scared ghost
            if scared_time == 0:
                if ghost_distance > 0:  # Eviting division by zero errors
                    score -= 2.0 / ghost_distance

            # We increase score as closer to the scared ghost
            elif scared_time > 0:
                if ghost_distance > 0:
                    score += 1.5 / ghost_distance


        # We decrease score for each food item left as fewer food is better
        score -= len(new_food.as_list())

        return score

def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.get_score()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, eval_fn='score_evaluation_function', depth='2'):
        super().__init__()
        self.index = 0 # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth) 

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action from the current game_state using self.depth
        and self.evaluation_function.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
        Returns a list of legal actions for an agent
        agent_index=0 means Pacman, ghosts are >= 1

        game_state.generate_successor(agent_index, action):
        Returns the successor game state after an agent takes an action

        game_state.get_num_agents():
        Returns the total number of agents in the game

        game_state.is_win():
        Returns whether or not the game state is a winning state

        game_state.is_lose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # We define a recursive function to calculate the minimax algorithm (both cases min and max)
        def minimax(game_state, depth, agent_index):

            # Bases cases: We check if the game finished or we reached the depth 0
            if game_state.is_win() or game_state.is_lose() or depth == 0:
                return self.evaluation_function(game_state)

            num_agents = game_state.get_num_agents()

            if agent_index == 0:  # Pacman turn (maximizing value)
                best_score = float('-inf')
                for action in game_state.get_legal_actions(agent_index):
                    successor_game_state = game_state.generate_successor(agent_index, action)
                    score = minimax(successor_game_state, depth, 1)
                    best_score = max(best_score, score)
                return best_score
            
            else:  # Ghost turn (minimizing value)
                best_score = float('inf')
                for action in game_state.get_legal_actions(agent_index):
                    successor_game_state = game_state.generate_successor(agent_index, action)
                    if agent_index == num_agents - 1:  # Last ghost
                        score = minimax(successor_game_state, depth - 1, 0)
                    else:
                        score = minimax(successor_game_state, depth, agent_index + 1)
                    best_score = min(best_score, score)
                return best_score
        
        # We now calculate the best action for Pacman
        best_action = None
        best_score = float('-inf')

        # We iterate over all legal actions
        for action in game_state.get_legal_actions(0):

            # We calculate the succesor state
            successor_game_state = game_state.generate_successor(0, action)

            # We calculate the score for the succesor state
            score = minimax(successor_game_state, self.depth, 1)

            # We update the best action and score
            if score > best_score:
                best_score = score
                best_action = action

        return best_action
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluation_function
        """
        "*** YOUR CODE HERE ***"
        def alpha_beta(game_state, depth, agent_index, alpha, beta):

            # Bases cases: We check if the game finished or we reached the depth 0
            if game_state.is_win() or game_state.is_lose() or depth == 0:
                return self.evaluation_function(game_state)

            num_agents = game_state.get_num_agents()

            if agent_index == 0:
                best_score = float('-inf')
                for action in game_state.get_legal_actions(agent_index):
                    successor_game_state = game_state.generate_successor(agent_index, action)
                    score = alpha_beta(successor_game_state, depth, 1, alpha, beta)
                    best_score = max(best_score, score)
                    alpha = max(alpha, best_score)
                    if beta < alpha:
                        break
                return best_score
            else:
                best_score = float('inf')
                for action in game_state.get_legal_actions(agent_index):
                    successor_game_state = game_state.generate_successor(agent_index, action)
                    if agent_index == num_agents - 1:
                        score = alpha_beta(successor_game_state, depth - 1, 0, alpha, beta)
                    else:
                        score = alpha_beta(successor_game_state, depth, agent_index + 1, alpha, beta)
                    best_score = min(best_score, score)
                    beta = min(beta, best_score)
                    if beta < alpha:
                        break
                return best_score

        best_action = None
        aplha = float('-inf')
        beta = float('inf')

        for action in game_state.get_legal_actions(0):      
            successor_game_state = game_state.generate_successor(0, action)
            score = alpha_beta(successor_game_state, self.depth, 1, aplha, beta)
            if score > aplha:
                aplha = score
                best_action = action
        
        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluation_function

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()

def better_evaluation_function(current_game_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()
    


# Abbreviation
better = better_evaluation_function
