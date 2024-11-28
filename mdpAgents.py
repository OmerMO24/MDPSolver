# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

# THIS IS THE FINAL ONE THAT I WILL BE SUBMITTING INSHALLAH

from enum import IntEnum
from pacman import SCARED_TIME, Directions
from game import Agent
import api
import random
import game
import util


class MDPAgent(Agent):
    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):
        print "Starting up MDPAgent!"

        # These Stay the same between the medium strategy and small strategy
        self.values = util.Counter()
        self.old_values = util.Counter()
        self.EPSILON = 0.01
        self.MAX_ITERS = 50
        self.WALLS = None
        self.CORNERS = None

        # Gamma is different for small grid and medium
        self.GAMMA = None


    def getAction(self, state):
        if self.WALLS is None:
            self.WALLS = api.walls(state)
            self.CORNERS = api.corners(state)

            # set gamma based on grid size
            width = max(x for x, y in self.CORNERS) + 1
            height = max(y for x, y in self.CORNERS) + 1
            self.GAMMA = 0.9 if width <= 7 and height <= 7 else 0.6

        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        if not legal:
            return Directions.STOP

        if self.GAMMA == 0.9:
            self.runValueIterationSmall(state)
        else:
            self.runValueIterationMedium(state)

        # Choose best action
        pacman_pos = api.whereAmI(state)
        max_value = float("-inf")
        best_action = legal[0]

        for action in legal:
            value = self.getQValue(state, pacman_pos, action)
            if value > max_value:
                max_value = value
                best_action = action

        return api.makeMove(best_action, legal)

    # SMALL STRATEGY

    def runValueIterationSmall(self, state):
        walls = self.WALLS
        corners = self.CORNERS
        width = max(x for x, y in corners) + 1
        height = max(y for x, y in corners) + 1

        iteration = 0
        while iteration < self.MAX_ITERS:
            self.old_values = self.values.copy()
            max_delta = 0

            for x in range(width):
                for y in range(height):
                    pos = (x, y)
                    if pos not in walls:
                        max_q = float("-inf")
                        legal = self.notWalls(pos)

                        for action in legal:
                            q_value = self.getQValue(state, pos, action)
                            max_q = max(max_q, q_value)

                        if max_q != float("inf"):
                            self.values[pos] = self.getReward(state, pos) + self.GAMMA * max_q
                            max_delta = max(max_delta, abs(self.values[pos] - self.old_values[pos]))

            if max_delta < self.EPSILON:
                break

            iteration += 1


    def getReward(self, state, pos):
        reward = 0
        remaining_food = len(api.food(state))

        # Food reward scaling - keep this strong
        if pos in api.food(state):
            food_reward = 10 * (100.0 / remaining_food) if remaining_food > 0 else 10
            reward += food_reward

        # Add distance-based food reward to guide Pacman
        food_positions = api.food(state)
        if food_positions:
            nearest_food_dist = min(util.manhattanDistance(pos, food_pos)
                                  for food_pos in food_positions)
            reward += 50 / (nearest_food_dist + 1)

        # Keep capsule reward moderate
        if pos in api.capsules(state):
            reward += 20

        # Reduce ghost rewards, keep penalties strong
        for ghost_pos, scared_time in api.ghostStatesWithTimes(state):
            dist = util.manhattanDistance(pos, ghost_pos)
            if scared_time > 0:  # Ghost is scared
                if dist < 2 and scared_time > dist + 2:
                    reward += 50  # Reduced from 100 - make ghost eating less attractive
            else:  # Ghost is dangerous
                if dist < 2:
                    reward -= 800  # Keep strong penalty
                elif dist < 4:
                    reward -= 400
                elif dist < 6:
                    reward -= 200 / (dist + 1)

        return reward


    ## MEDIUM STRATEGY

    def initializeValues(self, state):
        """Initialize terminal states (food) except those near ghosts"""
        self.values.clear()
        food_positions = api.food(state)
        remaining_food = len(food_positions)
        ghost_positions = {ghost_pos for ghost_pos, scared_time
                            in api.ghostStatesWithTimes(state) if not scared_time}

        # Scale terminal food values based on remaining food
        for food_pos in food_positions:
            # Higher reward when less food remains
            food_reward = 500 * (100.0 / remaining_food) if remaining_food > 0 else 500
            self.values[food_pos] = food_reward


    def runValueIterationMedium(self, state):
        self.initializeValues(state)

        corners = self.CORNERS
        width = max(x for x, y in corners) + 1
        height = max(y for x, y in corners) + 1
        food_positions = api.food(state)
        remaining_food = len(food_positions)
        capsules = api.capsules(state)

        ghost_radius = 4 # was 4 for medium. Note: 3 or 4 is good.
        squares_to_update = set()

        # Add empty squares
        for x in range(width):
            for y in range(height):
                pos = (x, y)
                if pos not in self.WALLS and pos not in food_positions:
                    squares_to_update.add(pos)

        # Add squares near ghosts (including food and capsules)
        for ghost_pos, scared_time in api.ghostStatesWithTimes(state):
            if not scared_time:
                for x in range(width):
                    for y in range(height):
                        pos = (x, y)
                        if (pos not in self.WALLS and
                            util.manhattanDistance(pos, ghost_pos) < ghost_radius):
                                squares_to_update.add(pos)

        iteration = 0
        while iteration < self.MAX_ITERS:
            self.old_values = self.values.copy()
            max_delta = 0

            for pos in squares_to_update:
                max_q = float("-inf")
                legal = self.notWalls(pos)

                for action in legal:
                    q_value = self.getQValue(state, pos, action)
                    max_q = max(max_q, q_value)

                if max_q != float("-inf"):
                    # Start with propagated value
                    self.values[pos] = self.GAMMA * max_q

                    # Add strong negative values for being near ghosts
                    for ghost_pos, scared_time in api.ghostStatesWithTimes(state):
                        if not scared_time:
                            dist = util.manhattanDistance(pos, ghost_pos)
                            if dist < ghost_radius:
                                if dist < 2:
                                    self.values[pos] -= 2000  # Very close to ghost
                                elif dist < 4:
                                    self.values[pos] -= 1000 / (dist + 1)  # Medium distance
                                else:
                                    self.values[pos] -= 500 / (dist + 1)  # Further away

                    # Add rewards for food/capsules after ghost penalties
                    if pos in food_positions:
                        food_reward = 100 * (100.0 / remaining_food) if remaining_food > 0 else 100
                        self.values[pos] += food_reward
                    elif pos in capsules:
                        self.values[pos] += 150

                    max_delta = max(max_delta, abs(self.values[pos] - self.old_values[pos]))

            if max_delta < self.EPSILON:
                break

            iteration += 1



    # HELPER FUNCTIONS THAT ARE GENERAL BETWEEN BOTH THE MEDIUM AND SMALL STRATEGY

    def getSuccessor(self, pos, action):
        x, y = pos
        if action == Directions.NORTH:
            return (x, y + 1)
        elif action == Directions.SOUTH:
            return (x, y - 1)
        elif action == Directions.EAST:
            return (x + 1, y)
        elif action == Directions.WEST:
            return (x - 1, y)
        return pos

    def getQValue(self, state, pos, action):
        intended_pos = self.getSuccessor(pos, action)
        if intended_pos not in self.WALLS and intended_pos in self.values:
            return self.values[intended_pos]
        return 0.0

    def notWalls(self, pos):
        legal = []
        x, y = pos

        if (x+1, y) not in self.WALLS:
            legal.append(Directions.EAST)
        if (x-1, y) not in self.WALLS:
            legal.append(Directions.WEST)
        if (x, y+1) not in self.WALLS:
            legal.append(Directions.NORTH)
        if (x, y-1) not in self.WALLS:
            legal.append(Directions.SOUTH)

        return legal

    def final(self, state):
        self.values.clear()
