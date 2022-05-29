from pacai.util import reflection
from pacai.agents.capture.offense import OffensiveReflexAgent
from pacai.agents.capture.defense import DefensiveReflexAgent
from pacai.core.directions import Directions
from pacai.agents.capture import CaptureAgent
import sys



# Feature Extractor(s) -------------------------------------------------------

class FeatureExtractor(abc.ABC):
    """
    A class that takes a `pacai.core.gamestate.AbstractGameState` and `pacai.core.actions.Actions`,
    and returns a dict of features.
    """

    @abc.abstractmethod
    def getFeatures(self, state, action):
        """
        Returns a dict from features to counts
        Usually, the count will just be 1.0 for
        indicator functions.
        """

        pass

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman.
    """

    def getFeatures(self, state, action):
        # Extract the grid of food and wall locations and get the ghost locations.
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = {}
        features["bias"] = 1.0

        # Compute the location of pacman after he takes the action.
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # Count the number of ghosts 1-step away.
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in
                Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # If there is no danger of ghosts then add the food feature.
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        prob = AnyFoodSearchProblem(state, start = (next_x, next_y))
        dist = len(search.bfs(prob))
        if dist is not None:
            # Make the distance a number less than one otherwise the update will diverge wildly.
            features["closest-food"] = float(dist) / (walls.getWidth() * walls.getHeight())

        for key in features:
            features[key] /= 10.0

        return features










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
        walls = gameState.getWalls

        # Compute distance to the nearest food.
        foodList = self.getFood(gameState).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDist = min([self.getMazeDistance(myPos, food) for food in foodList])
            # MazeDist problem: doesn't return '0' if food on the successor state, thus,
            if(myPos in foodList):
                features['distanceToFoodRecip'] = 1
                # print("HERE")
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

# Modal Agent ----------------------------------------------------------------

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

        self.currentAgent = 'offenseReflexAgent'

    def registerInitialState(self, gameState):
        super().registerInitialState(self, gameState)

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

        # Update each agent and get its action
        actions = {}
        for agent in self.agents:
            actions[agent] = self.agents[agent].chooseAction(gameState)

        # self.rudeStrat(gameState)

        return actions[self.currentAgent]

# Create Team ----------------------------------------------------------------

def createTeam(firstIndex, secondIndex, isRed,
    first = 'pacai.agents.capture.dummy.DummyAgent',
    second = 'pacai.agents.capture.dummy.DummyAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    # firstAgent = reflection.qualifiedImport(first)
    # secondAgent = reflection.qualifiedImport(second)

    # firstAgent = ModalAgent
    # secondAgent = ModalAgent

    firstAgent = OffensiveAgent
    secondAgent = DefensiveAgent

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]



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