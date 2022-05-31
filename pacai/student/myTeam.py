from pacai.util import reflection
from pacai.agents.capture.offense import OffensiveReflexAgent
from pacai.agents.capture.defense import DefensiveReflexAgent
from pacai.agents.capture.capture import CaptureAgent
from pacai.core.directions import Directions
import sys
import time

# Q-Learning
from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.util import probability
import random
import pacai.bin.capture
from pacai.core.actions import Actions
import abc
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search import search

# Reflex Agents --------------------------------------------------------------------------------

class OffensiveAgent(OffensiveReflexAgent):
    def __init__(self, index):
        super().__init__(index)

    def getFeatures(self, gameState, action):
        """
        FEATURES:
        onOffense: being on the offense is prioritized.

        """
        features = {}
        successor = self.getSuccessor(gameState, action)  # gets successor state if this action taken
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        if(myState.isPacman()):
            features['onOffense'] = 1
        else:
            features['onOffense'] = 0

        # Compute distance to the ghosts we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman() and a.getPosition() is not None]
        numGhosts = len(ghosts)

        if(action == Directions.STOP):
            features['stop'] = 1
        else:
            features['stop'] = 0

        if(numGhosts > 0):
            dists = []
            # closestGhost = ghosts[0]
            for a in ghosts:
                dists.append(self.getMazeDistance(myPos, a.getPosition()))
            # closestGhost = dists.
            minDist = min(dists)
            if(minDist == 0):
                # Don't divide by 0
                features['enemyDistanceRecip'] = sys.maxsize
            else:
                features['enemyDistanceRecip'] = 1/(minDist * minDist)


        # Compute distance to the nearest food.
        foodList = self.getFood(gameState).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDist = min([self.getMazeDistance(myPos, food) for food in foodList])
            # MazeDist problem: doesn't return '0' if food on the successor state, thus,
            if(myPos in foodList):
                features['distanceToFoodRecip'] = 1
            else:
                features['distanceToFoodRecip'] = 1/(minDist)

        # Deprioritize reversing
        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {
            'reverse': -5,
            'stop': -100,
            'onOffense': 10,
            'enemyDistanceRecip': -30,
            'distanceToFoodRecip': 10,
        }

class DefensiveAgent(DefensiveReflexAgent):
    def __init__(self, index):
        super().__init__(index)

    def getFeatures(self, gameState, action):
        features = {}

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 1000,
            'invaderDistance': -1000,
            'stop': -100,
            'reverse': -2
        }

# Feature Extractor(s) -------------------------------------------------------

# TODO: zombie (probably)
class AnyFoodSearchProblem(PositionSearchProblem):

    def __init__(self, gameState, start = None):
        super().__init__(gameState, goal = None, start = start)

        # Store the food for later reference.
        self.food = gameState.getFood()

    def isGoal(self, state):
        return state in self.food.asList()

# TODO: probably unnecessary
"""
class OpponentFoodSearchProblem(PositionSearchProblem):

    def __init__(self, gameState, start=None):
        super().__init__(gameState, goal = None, start = start)
"""

# Q-LEARNING ------------------------------------------------------------------------------------

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

        The core of this class is a qValues dictionary. This stores the
        (state, action) q values. getQValue(state, action) returns values from this dictionary.
        getValue() calculates the value of the best action in a state. getPolicy() returns the
        best action from a state. update()  calculates new q-values given a sample (s, a, s', r).
        getAction() returns the best action, or a random action epsilon amount of the time.
    """

    def __init__(self, index, **kwargs):
        actionFn = lambda state: state.getLegalActions(index)
        super().__init__(index, actionFn, **kwargs)

        # A dictionary of (state, action) keys where the value is the q value
        self.qValues = {}

    # OVERRIDEN
    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """

        if (state, action) in self.qValues:
            return self.qValues[(state, action)]
        else:
            return 0.0

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """
        actions = self.getLegalActions(state)
        if not len(actions):
            return 0
        maxQ = self.getQValue(state, actions[0])
        for action in actions:
            if (state, action) in self.qValues:
                qVal = self.getQValue(state, action)
                if qVal > maxQ:
                    maxQ = qVal
        return maxQ

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        actions = self.getLegalActions(state)
        if not len(actions):
            return None

        maxQ = self.getQValue(state, actions[0])
        maxActions = [actions[0]]
        for action in actions:
            qVal = self.getQValue(state, action)
            if qVal == maxQ:
                if action not in maxActions:
                    maxActions.append(action)
            elif qVal > maxQ:
                maxQ = qVal
                maxActions = [action]

        maxAction = random.choice(maxActions)
        return maxAction

    def update(self, state, action, nextState, reward):
        """
        Implementation of q-learning.
        """
        # print("reward: ", reward)
        alpha = self.alpha
        qOld = self.getQValue(state, action)
        discount = self.discountRate
        vNext = self.getValue(nextState)
        sample = reward + (discount * vNext)
        qNew = ((1 - alpha) * qOld) + (alpha * sample)
        self.qValues[(state, action)] = qNew
        # print("new q:", qNew)
        return

    def getQAction(self, state):
        """
        Return the action to be taken from state. This chooses random actions epsilon percent
        of the time.
        """
        takeRandAction = probability.flipCoin(self.epsilon)
        if takeRandAction:
            action = random.choice(self.getLegalActions(state))
            return action
        else:
            return self.getPolicy(state)


class ApproximateQCaptureAgent(CaptureAgent, QLearningAgent):
    """
    Subclass of QlearningAgent which turns it into an approximate Capture Agent.

    TODO: consider removing STOP direction possibility for offense.
    NOTE: if we want a custom reward function we can overwrite observationFunction() (it's
          what calls update())
    """
    def __init__(self, index, numTraining=0, **kwargs):
        # TODO: make sure that learning parameters get turned off in competition

        # Training/learning parameters
        self.l_epsilon = 0.05
        self.l_gamma = 0.08
        self.l_alpha = 0.1

        # Competition/testing parameters
        self.c_epsilon = 0  # Don't take random actions
        self.c_gamma = 0.05   # Going to keep learning, though slower
        self.c_alpha = 0.1

        CaptureAgent.__init__(self, index, **kwargs)
        QLearningAgent.__init__(self, index, **kwargs)

        self.numTraining = numTraining

        if(self.isInTraining()):
            self.epsilon = self.l_epsilon
            self.gamma = self.l_gamma
            self.alpha = self.l_alpha
        else:
            self.epsilon = self.c_epsilon
            self.gamma = self.c_gamma
            self.alpha = self.c_alpha


        # Initial weights
        self.weights = self.getInitialWeights()

        """
        # DEPRECATED
        # Training
        self.trainingAgent = False
        if('trainingAgent' in kwargs):
            self.trainingAgent = True

        self.trainingBlueTeamPath = 'pacai.student.myTeam'
        self.trainingRedTeamPath = 'pacai.student.myTeam'
        self.trainingArgv = ['--red', self.trainingBlueTeamPath,
                             '--red-args', 'trainingAgent=true',
                             '--blue', self.trainingRedTeamPath,
                             '--blue-args', 'trainingAgent=true',
                             '--num-training', str(numTraining),
                             '--num-games', str(numTraining + 1),  # crashes without +1
                             '--null-graphics',
                             '-q']
        """


    def registerInitialState(self, gameState):
        """
        We have 15 seconds to do whatever we want before each round. This is
        where we will do our training. Training ends when this function
        finishes.
        """
        QLearningAgent.registerInitialState(self, gameState)
        CaptureAgent.registerInitialState(self, gameState)

        # Training
        # TODO: learn about food (override this method)

        """
        # DEPRECATED
        # NOTE: I think each agent gets 15 seconds.
        if not self.trainingAgent:
            if(self.numTraining > 0):
                print("running capture main, num_training: ", self.numTraining)
                startTime = time.perf_counter()
                pacai.bin.capture.main(self.trainingArgv)  # ~1sec for one round
                endTime = time.perf_counter()
                print("time to run one capture game: ", endTime-startTime)
                # TODO: parse stdout weights
        """


    # Overriden
    def getFeatures(self, state, action):
        """
        Returns a dict from features to counts
        Usually, the count will just be 1.0 for
        indicator functions.
        """
        features = {}
        features["bias"] = 1
        return features

    # Overriden
    def getInitialWeights(self):
        """
        Return weights which have been picked by hand. Better this than
        starting with a completely zeroed out dictionary.
        """
        return {"bias": 0}

    def chooseAction(self, gameState):
        action = self.getQAction(gameState)
        return action

    def getQAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getQAction(state)
        self.doAction(state, action)

        return action

    def getQValue(self, state, action):
        # Get feature vector (dictionary)
        features = self.getFeatures(state, action)

        # Take dot product of w and feature vector
        q = 0
        for feature in features:
            if feature not in self.weights:
                self.weights[feature] = 0.0
            q += features[feature] * self.weights[feature]

        return q

    def update(self, state, action, nextState, reward):
        alpha = self.alpha
        discount = self.discountRate
        qOld = self.getQValue(state, action)
        vNext = self.getValue(nextState)

        correction = (reward + (discount * vNext)) - qOld

        # Update weights
        features = self.getFeatures(state, action)
        for feature in features:
            if feature not in self.weights:
                oldWeight = 0.0
            else:
                oldWeight = self.weights[feature]
            self.weights[feature] = oldWeight + (alpha * correction * features[feature])


    def final(self, state):
        """
        Called at the end of each game.
        Note: there is a much more thorough final function in Reinforcement, it's called
        first here.
        """

        # Call the Reinforcement final method.
        QLearningAgent.final(self, state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:

            # Print out weight dictionary
            print("Weights: ", self.weights)

            # Set parameters to competition/testing
            self.epsilon = self.c_epsilon
            self.gamma = self.c_gamma
            self.alpha = self.c_alpha


class OffensiveQAgent(ApproximateQCaptureAgent):
    def __init__(self, index, numTraining, **kwargs):
        ApproximateQCaptureAgent.__init__(self, index, numTraining, **kwargs)

    def observationFunction(self, state):
        """
        Custom reward function. Called after each move.
        """

        if self.lastState is not None:
            reward = 0

            # Reward for eating foodj
            foodEaten = -(len(self.getFood(state).asList()) - len(self.getFood(self.lastState).asList()))
            if(foodEaten):
                reward += 10
            else:
                reward += -0.1
            print(reward)

            self.observeTransition(self.lastState, self.lastAction, state, reward)

    def getNextPosition(self, state, action):
        """
        Return next_x, next_y tuple.
        """
        # Compute the location of our agent after it takes the action.
        x, y = state.getAgentPosition(self.index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        return (next_x, next_y)

    def getFeatures(self, state, action):
        foodList = self.getFood(state).asList()
        walls = state.getWalls()

        # This calculates the entire game state if the action is taken (opponents don't move).
        # Be careful with this. For one, it might be take lots of processing (need to look closer).
        #                       It also calculates whether food eaten or not, so make sure you don't
        #                       use this to find closes food
        successor = state.generateSuccessor(self.index, action)  # gets successor state if this action taken
        agentSuccState = successor.getAgentState(self.index)
        nextPos = agentSuccState.getPosition()

        features = {}
        features["bias"] = 1.0

        # Feature: whether next state is on offense or not
        if (agentSuccState.isPacman()):
            features['onOffense'] = 1
            features['onDefense'] = 0
        else:
            features['onOffense'] = 0
            features['onDefense'] = 1


        if(action == Directions.STOP):
            features['stop'] = 1
        else:
            features['stop'] = 0

        # nextPos = self.getNextPosition(state, action)  # can use this instead if we abandon generateSuccessor

        # Feature: reciprocal of distance to the closest ghosts, TODO
        # Compute distance to the ghosts we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman() and a.getPosition() is not None]
        numGhosts = len(ghosts)
        if (numGhosts > 0):
            dists = []
            for a in ghosts:
                dists.append(self.getMazeDistance(nextPos, a.getPosition()))
            minDist = min(dists)
            features['enemyDistanceRecip'] = 1 / minDist if minDist != 0 else 1

        # Feature: closest food reciprocal
        closestFood = min([self.getMazeDistance(nextPos, food) for food in foodList])
        features['closest-food'] = 1/closestFood if closestFood != 0 else 1

        # Feature: Deprioritize reversing
        rev = Directions.REVERSE[agentSuccState.getDirection()]
        if (action == rev):
            features['reverse'] = 1

        """
        prob = AnyFoodSearchProblem(state, start=nextPos)
        dist = len(search.bfs(prob))
        if dist is not None:
            # Make the distance a number less than one otherwise the update will diverge wildly.
            features["closest-food"] = float(dist) / (walls.getWidth() * walls.getHeight())
        """

        return features

    def getInitialWeights(self):
        return {
            'reverse': 0,  #-20
            'stop': 0,  #-100
            'onOffense': 10,  #10
            'onDefense': -10,
            'enemyDistanceRecip': -1,  #-30
            'closest-food': 10,  #10
        }



# TODO
# class DefensiveQAgent(ApproximateQCaptureAgent)

# Modal Agent ---------------------------------------------------------------------------------

class ModalAgent(CaptureAgent):
    """
    This agent creates and hosts multiple agents at the same time. The purpose of this is
    to switch between these different agents like they are different modes. It is necessary
    to run agents in parallel because some agents (like q-learning) require constant information.
    """
    def __init__(self, index):
        super().__init__(index)

        # TODO: make sure both modalAgents aren't identical (i.e. one making the same exact
        # moves as the other)
        # Possible solution: have 2 defense agents, each one tracking one individual opponent.

        # Create agents
        self.agents = {
            'offenseReflexAgent': OffensiveAgent(index),
            'defenseReflexAgent': DefensiveAgent(index)
        }

        # TODO: replace this with the actual way supposed to do (dummy)
        self.initAgents = False

        self.currentAgent = 'offenseReflexAgent'

    def switchMode(self, newMode):
        """
        This method switches the agent into a new mode.
        """
        self.currentMode = newMode

    def rudeStrat(self, gameState):
        """
        This performs the necessary switching of agent modes in order accordance
        with the rude strategy.
        """
        # Check to see where opponents are..
        # Switch to offense..
        # Back to defense...
        # etc
        pass

    def chooseAction(self, gameState):
        # Need to initialize agents at startup
        if not self.initAgents:
            for agent in self.agents:
                self.agents[agent].registerInitialState(gameState)
            self.initAgents = True

        # Update each agent and get its action
        actions = {}
        for agent in self.agents:
            actions[agent] = self.agents[agent].chooseAction(gameState)

        # self.rudeStrat(self, gameState)

        return actions[self.currentAgent]

# CreateTeam ------------------------------------------------------------------------------------

def createTeam(firstIndex, secondIndex, isRed,
               numTraining = 0,
               first='pacai.agents.capture.dummy.DummyAgent',
               second='pacai.agents.capture.dummy.DummyAgent',
               **kwargs
               ):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    # Baseline
    # firstAgent = reflection.qualifiedImport(first)
    # secondAgent = reflection.qualifiedImport(second)

    # Modal
    #secondAgent = ModalAgent()

    #numTraining = 5
    firstAgent = OffensiveQAgent(firstIndex, numTraining, **kwargs)
    # secondAgent = ApproximateQCaptureAgent(secondIndex, **kwargs)
    secondAgent = DefensiveReflexAgent(secondIndex)

    # Reflex
    #firstAgent = OffensiveAgent(firstIndex)
    #secondAgent = DefensiveAgent(secondIndex)

    return [
        firstAgent,
        secondAgent
    ]

# from pacai.util import reflection
# from pacai.agents.capture.offense import OffensiveReflexAgent
# from pacai.agents.capture.defense import DefensiveReflexAgent
# from pacai.core.directions import Directions
# from pacai.agents.capture.capture import CaptureAgent
# import sys
# import abc

# from pacai.core.search.position import PositionSearchProblem



# # Feature Extractor(s) -------------------------------------------------------

# class FeatureExtractor(abc.ABC):
#     """
#     A class that takes a `pacai.core.gamestate.AbstractGameState` and `pacai.core.actions.Actions`,
#     and returns a dict of features.
#     """

#     @abc.abstractmethod
#     def getFeatures(self, state, action):
#         """
#         Returns a dict from features to counts
#         Usually, the count will just be 1.0 for
#         indicator functions.
#         """

#         pass


# class AnyFoodSearchProblem(PositionSearchProblem):
#     """
#     Required by SimpleExtractor.
#     """


#     def __init__(self, gameState, start = None):
#         super().__init__(gameState, goal = None, start = start)

#         # Store the food for later reference.
#         self.food = gameState.getFood()

#     def isGoal(self, state):
#         return state in self.food.asList()


# class SimpleExtractor(FeatureExtractor):
#     """
#     Returns simple features for a basic reflex Pacman.
#     """

#     def getFeatures(self, state, action):
#         # Extract the grid of food and wall locations and get the ghost locations.
#         food = state.getFood()
#         walls = state.getWalls()
#         ghosts = state.getGhostPositions()

#         features = {}
#         features["bias"] = 1.0

#         # Compute the location of pacman after he takes the action.
#         x, y = state.getPacmanPosition()
#         dx, dy = Actions.directionToVector(action)
#         next_x, next_y = int(x + dx), int(y + dy)

#         # Count the number of ghosts 1-step away.
#         features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in
#                 Actions.getLegalNeighbors(g, walls) for g in ghosts)

#         # If there is no danger of ghosts then add the food feature.
#         if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
#             features["eats-food"] = 1.0

#         prob = AnyFoodSearchProblem(state, start = (next_x, next_y))
#         dist = len(search.bfs(prob))
#         if dist is not None:
#             # Make the distance a number less than one otherwise the update will diverge wildly.
#             features["closest-food"] = float(dist) / (walls.getWidth() * walls.getHeight())

#         for key in features:
#             features[key] /= 10.0

#         return features



# # Agents ---------------------------------------------------------------------

# class OffensiveAgent(OffensiveReflexAgent):
#     def __init__(self, index):
#         super().__init__(index)

#     def getFeatures(self, gameState, action):
#         """
#         FEATURES:
#         onOffense: being on the offense is prioritized.

#         """
#         features = {}
#         successor = self.getSuccessor(gameState, action)  # gets successor state if this action taken
#         myState = successor.getAgentState(self.index)
#         myPos = myState.getPosition()
#         if(myState.isPacman()):
#             features['onOffense'] = 1
#         else:
#             features['onOffense'] = 0

#         # Compute distance to the ghosts we can see
#         enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
#         ghosts = [a for a in enemies if not a.isPacman() and a.getPosition() is not None]
#         numGhosts = len(ghosts)

#         if(action == Directions.STOP):
#             features['stop'] = 1
#         else:
#             features['stop'] = 0

#         if(numGhosts > 0):
#             dists = []
#             # closestGhost = ghosts[0]
#             for a in ghosts:
#                 dists.append(self.getMazeDistance(myPos, a.getPosition()))
#             # closestGhost = dists.
#             minDist = min(dists)
#             if(minDist == 0):
#                 # Don't divide by 0
#                 features['enemyDistanceRecip'] = sys.maxsize
#             else:
#                 features['enemyDistanceRecip'] = 1/(minDist * minDist)
#         walls = gameState.getWalls

#         # Compute distance to the nearest food.
#         foodList = self.getFood(gameState).asList()

#         # This should always be True, but better safe than sorry.
#         if (len(foodList) > 0):
#             myPos = successor.getAgentState(self.index).getPosition()
#             minDist = min([self.getMazeDistance(myPos, food) for food in foodList])
#             # MazeDist problem: doesn't return '0' if food on the successor state, thus,
#             if(myPos in foodList):
#                 features['distanceToFoodRecip'] = 1
#                 # print("HERE")
#             else:
#                 features['distanceToFoodRecip'] = 1/(minDist)

#         # Deprioritize reversing
#         rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
#         if (action == rev):
#             features['reverse'] = 1

#         return features

#     def getWeights(self, gameState, action):
#         return {
#             'reverse': -5,
#             'stop': -100,
#             'onOffense': 10,
#             'enemyDistanceRecip': -30,
#             'distanceToFoodRecip': 10,
#         }

# class DefensiveAgent(DefensiveReflexAgent):
#     def __init__(self, index):
#         super().__init__(index)

#     def getFeatures(self, gameState, action):
#         features = {}

#         successor = self.getSuccessor(gameState, action)
#         myState = successor.getAgentState(self.index)
#         myPos = myState.getPosition()

#         # Computes whether we're on defense (1) or offense (0).
#         features['onDefense'] = 1
#         if (myState.isPacman()):
#             features['onDefense'] = 0

#         # Computes distance to invaders we can see.
#         enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
#         invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
#         features['numInvaders'] = len(invaders)

#         if (len(invaders) > 0):
#             dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
#             features['invaderDistance'] = min(dists)

#         if (action == Directions.STOP):
#             features['stop'] = 1

#         rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
#         if (action == rev):
#             features['reverse'] = 1

#         return features

#     def getWeights(self, gameState, action):
#         return {
#             'numInvaders': -1000,
#             'onDefense': 1000,
#             'invaderDistance': -1000,
#             'stop': -100,
#             'reverse': -2
#         }

# # Modal Agent ----------------------------------------------------------------

# class ModalAgent(CaptureAgent):
#     """
#     This agent creates and hosts multiple agents at the same time. The purpose of this is
#     to switch between these different agents like they are different modes. It is necessary
#     to run agents in parallel because some agents (like q-learning) require constant information.
#     """
#     def __init__(self, index):
#         super().__init__(index)

#         # TODO: make sure both modalAgents aren't identical (i.e. one making the same exact
#         # moves as the other)
#         # Possible solution: have 2 defense agents, each one tracking one individual opponent.

#         # Create agents
#         self.agents = {
#             'offenseReflexAgent': OffensiveAgent(index),
#             'defenseReflexAgent': DefensiveAgent(index)
#         }

#         self.currentAgent = 'offenseReflexAgent'

#     def registerInitialState(self, gameState):
#         super().registerInitialState(self, gameState)

#     def switchMode(self, newMode):
#         """
#         This method switches the agent into a new mode.
#         """
#         self.currentMode = newMode

#     def rudeStrat(self, gameState):
#         """
#         This performs the necessary switching of agent modes in order accordance
#         with the rude strategy.
#         """
#         # Check to see where opponents are..
#         # Switch to offense..
#         # Back to defense...
#         # etc
#         pass

#     def chooseAction(self, gameState):

#         # Update each agent and get its action
#         actions = {}
#         for agent in self.agents:
#             actions[agent] = self.agents[agent].chooseAction(gameState)

#         # self.rudeStrat(gameState)

#         return actions[self.currentAgent]

# # Create Team ----------------------------------------------------------------

# def createTeam(firstIndex, secondIndex, isRed,
#     first = 'pacai.agents.capture.dummy.DummyAgent',
#     second = 'pacai.agents.capture.dummy.DummyAgent'):
#     """
#     This function should return a list of two agents that will form the capture team,
#     initialized using firstIndex and secondIndex as their agent indexed.
#     isRed is True if the red team is being created,
#     and will be False if the blue team is being created.
#     """

#     # firstAgent = reflection.qualifiedImport(first)
#     # secondAgent = reflection.qualifiedImport(second)

#     # firstAgent = ModalAgent
#     # secondAgent = ModalAgent

#     firstAgent = OffensiveAgent
#     secondAgent = DefensiveAgent

#     return [
#         firstAgent(firstIndex),
#         secondAgent(secondIndex),
#     ]



# from pacai.util import reflection
# from pacai.agents.capture.offense import OffensiveReflexAgent
# from pacai.agents.capture.defense import DefensiveReflexAgent
# from pacai.core.directions import Directions
# import sys

# class OffensiveAgent(OffensiveReflexAgent):
#     def __init__(self, index):
#         super().__init__(index)

#     def getFeatures(self, gameState, action):
#         """
#         FEATURES:
#         onOffense: being on the offense is prioritized.

#         """
#         features = {}
#         successor = self.getSuccessor(gameState, action)  # gets successor state if this action taken
#         myState = successor.getAgentState(self.index)
#         myPos = myState.getPosition()
#         if(myState.isPacman()):
#             features['onOffense'] = 1
#         else:
#             features['onOffense'] = 0

#         # Compute distance to the ghosts we can see
#         enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
#         ghosts = [a for a in enemies if not a.isPacman() and a.getPosition() is not None]
#         numGhosts = len(ghosts)

#         if(action == Directions.STOP):
#             features['stop'] = 1
#         else:
#             features['stop'] = 0

#         if(numGhosts > 0):
#             dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
#             minDist = min(dists)
#             if(minDist == 0):
#                 # Don't divide by 0
#                 features['enemyDistanceRecip'] = sys.maxsize
#             else:
#                 features['enemyDistanceRecip'] = 1/(minDist * minDist)


#         # Compute distance to the nearest food.
#         foodList = self.getFood(gameState).asList()

#         # This should always be True, but better safe than sorry.
#         if (len(foodList) > 0):
#             myPos = successor.getAgentState(self.index).getPosition()
#             minDist = min([self.getMazeDistance(myPos, food) for food in foodList])
#             # MazeDist problem: doesn't return '0' if food on the successor state, thus,
#             if(myPos in foodList):
#                 features['distanceToFoodRecip'] = 1
#                 # print("HERE")
#             else:
#                 features['distanceToFoodRecip'] = 1/(minDist)

#         return features

#     def getWeights(self, gameState, action):
#         return {
#             'stop': -100,
#             'onOffense': 10,
#             'enemyDistanceRecip': -30,
#             'distanceToFoodRecip': 10,
#         }

# class DefensiveAgent(DefensiveReflexAgent):
#     def __init__(self, index):
#         super().__init__(index)

# def createTeam(firstIndex, secondIndex, isRed,
#     first = 'pacai.agents.capture.dummy.DummyAgent',
#     second = 'pacai.agents.capture.dummy.DummyAgent'):
#     """
#     This function should return a list of two agents that will form the capture team,
#     initialized using firstIndex and secondIndex as their agent indexed.
#     isRed is True if the red team is being created,
#     and will be False if the blue team is being created.
#     """

#     # firstAgent = reflection.qualifiedImport(first)
#     # secondAgent = reflection.qualifiedImport(second)

#     firstAgent = OffensiveAgent
#     secondAgent = DefensiveAgent

#     return [
#         firstAgent(firstIndex),
#         secondAgent(secondIndex),
#     ]






# from pacai.util import reflection
# from pacai.agents.capture.reflex import ReflexCaptureAgent

# def createTeam(firstIndex, secondIndex, isRed,
#         first = 'pacai.student.myTeam.OffensiveReflexAgent',
#         second = 'pacai.agents.capture.defense.DefensiveReflexAgent'):
#     """
#     This function should return a list of two agents that will form the capture team,
#     initialized using firstIndex and secondIndex as their agent indexed.
#     isRed is True if the red team is being created,
#     and will be False if the blue team is being created.
#     """

#     firstAgent = reflection.qualifiedImport(first)
#     secondAgent = reflection.qualifiedImport(second)

#     return [
#         firstAgent(firstIndex),
#         secondAgent(secondIndex),
#     ]


# class OffensiveReflexAgent(ReflexCaptureAgent):
#     """
#     A reflex agent that seeks food.
#     This agent will give you an idea of what an offensive agent might look like,
#     but it is by no means the best or only way to build an offensive agent.
#     """

#     def __init__(self, index, **kwargs):
#         super().__init__(index)

#     def getFeatures(self, gameState, action):
#         features = {}
#         successor = self.getSuccessor(gameState, action)
#         features['successorScore'] = self.getScore(successor)

#         # Compute distance to the nearest food.
#         foodList = self.getFood(successor).asList()

#         # This should always be True, but better safe than sorry.
#         if (len(foodList) > 0):
#             myPos = successor.getAgentState(self.index).getPosition()
#             minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
#             features['distanceToFood'] = minDistance

#         enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
#         defenders = [a for a in enemies if a.isGhost() and a.getPosition() is not None]
#         # features['numDefenders'] = len(defenders)

#         if (len(defenders) > 0):
#             dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
#             features['defenderDistance'] = min(dists)

#         return features

#     def getWeights(self, gameState, action):
#         return {
#             'successorScore': 100,
#             'distanceToFood': -1,
#             'defenderDistance': 10

#         }