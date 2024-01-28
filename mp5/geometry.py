# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import math
import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """   
    for wall in walls:
        wall = (wall[0], wall[1]), (wall[2], wall[3])
        if alien.is_circle():
            if point_segment_distance(alien.get_centroid(), wall) <= alien.get_width():
                return True
        else:
            head, tail = alien.get_head_and_tail()
            if point_segment_distance(head, wall) <= alien.get_width() or \
                point_segment_distance(tail, wall) <= alien.get_width():
                return True
            if segment_distance(alien.get_head_and_tail(), wall) <= alien.get_width():
                return True

    return False


def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    walls = [(0, 0, 0, window[0]), \
            (0, window[1], window[0], window[1]), \
            (0, 0, window[0], 0), \
            (window[0], 0, window[0], window[1])]

    return not does_alien_touch_wall(alien, walls)

def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """
    for wall in walls:
        wall = (wall[0], wall[1]), (wall[2], wall[3])
        if alien.is_circle():
            if point_segment_distance(alien.get_centroid(), wall) <= alien.get_width() or \
                point_segment_distance(waypoint, wall) <= alien.get_width():
                return True
            if segment_distance((alien.get_centroid(), waypoint), wall) <= alien.get_width():
                return True        
        else:
            centroid = alien.get_centroid()
            head, tail = alien.get_head_and_tail()
            waypoint_head = vector_add(waypoint, vector(centroid, head))
            waypoint_tail = vector_add(waypoint, vector(centroid, tail))
            polygon = head, tail, waypoint_head, waypoint_tail
            if is_point_in_polygon(wall[0], polygon) or is_point_in_polygon(wall[1], polygon):
                return True
            if segment_distance((head, tail), wall) <= alien.get_width() or \
            segment_distance((head, waypoint_head), wall) <= alien.get_width() or \
            segment_distance((waypoint_head, waypoint_tail), wall) <= alien.get_width() or \
            segment_distance((waypoint_tail, tail), wall) <= alien.get_width():
                return True

    return False

def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """
    edges = [(polygon[0], polygon[1]), (polygon[1], polygon[2]), \
             (polygon[2], polygon[3]), (polygon[3], polygon[0])]
    for edge in edges:
        if point_segment_distance(point, edge) == 0:
            return True    
    for edge in edges:
        if cross(vector(edge[0], edge[1]), vector(edge[0], point)) <= 0:
            return False
    return True

def vector(p, s):
    """Compute the vector from the two given points.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple (x, y) of the coordinates of the point.

        Return:
            The vector that s points to p.
    """
    return s[0] - p[0], s[1] - p[1]

def vector_add(p, s):
    return s[0] + p[0], s[1] + p[1]

def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]

def magnitude(v):
    ret = 0
    for x in v:
        ret += x * x
    return math.sqrt(ret)

def dot(a, b):
    ret = 0
    for i in range(len(a)):
        ret += a[i] * b[i]
    return ret

def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    ps0 = vector(p, s[0])
    ps1 = vector(p, s[1])
    s0s1 = vector(s[0], s[1])
    if magnitude(s0s1) == 0:
        return magnitude(ps0)
    point_line_distance = abs(cross(ps0, s0s1) / magnitude(s0s1))
    if -dot(ps0, s0s1) > 0 and dot(ps1, s0s1) > 0:
        # dot s0p, s0s1 > 0 means angle p, s0, s1 is acute
        return point_line_distance
    else:
        # p project to s is not on s
        return min(magnitude(ps0), magnitude(ps1))

def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    if point_segment_distance(s1[0], s2) == 0 or point_segment_distance(s1[1], s2) == 0 or \
        point_segment_distance(s2[0], s1) == 0 or point_segment_distance(s2[1], s1) == 0:
        return True

    return cross(vector(s1[0], s2[0]), vector(s1[0], s2[1])) * \
        cross(vector(s1[1], s2[0]), vector(s1[1], s2[1])) < 0 and \
        cross(vector(s2[0], s1[0]), vector(s2[0], s1[1])) * \
        cross(vector(s2[1], s1[0]), vector(s2[1], s1[1])) < 0

def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if do_segments_intersect(s1, s2):
        return 0
    else:
        return min(point_segment_distance(s1[0], s2), point_segment_distance(s1[1], s2), \
                   point_segment_distance(s2[0], s1), point_segment_distance(s2[1], s1))


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        walls = [(210, 120, 210, 130)]
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    # test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    # test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    """"
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)
    """
    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_horz, (207, 154), [True], [(232, 118)])
    """
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls

    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)
    """
    """
    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])
    """
    # Edge case coincide line endpoints
    """
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))
    """
    print("Geometry tests passed\n")
