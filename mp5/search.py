# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from state import MazeState
from collections import deque
import heapq


# Search should return the path and the number of states explored.
# The path should be a list of MazeState objects that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (astar)
# You may need to slight change your previous search functions in MP2 since this is 3-d maze


def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)


# TODO: VI
def astar(maze):
    starting_state = maze.get_start()
    visited_states = {starting_state: (None, starting_state)}
    frontier = []
    heapq.heappush(frontier, starting_state)
    done = set()
    
    while len(frontier) > 0:
        now_state = heapq.heappop(frontier)
        if now_state in done:
            continue
        done.add(now_state)
        if now_state.is_goal():
            return backtrack(visited_states, now_state)
        # print(now_state, "'s neighbors:", now_state.get_neighbors())
        for neighbor in now_state.get_neighbors():
            if neighbor in done:
                continue
            if neighbor not in visited_states or neighbor < visited_states[neighbor][1]:
                visited_states[neighbor] = (now_state, neighbor)
                heapq.heappush(frontier, neighbor)

    return None

# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
# TODO: VI
def backtrack(visited_states, goal_state):
    path = []
    # Your code here ---------------
    now_state = goal_state
    while True:
        path = [now_state] + path
        if visited_states[now_state][0]:
            now_state = visited_states[now_state][0]
        else:
            break
    # ------------------------------
    return path