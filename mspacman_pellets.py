"""Extract a coarse direction to an uneaten pellet in the original ALE maze."""

from heapq import heappop, heappush

import cv2
import numpy as np


NO_DIRECTION = 4
WALL_COLOUR = (228, 111, 111)
# RAM bit layout and screen coordinates follow OCAtari's MsPacman extractor.
# https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/mspacman.py#L237-L286
PELLET_BITS = (
    (0, 6), (1, 7), (1, 5), (1, 3), (1, 1), (2, 0), (2, 2), (2, 4), (2, 6),
    (1, 6), (1, 4), (1, 2), (1, 0), (2, 1), (2, 3), (2, 5), (2, 7), (0, 4),
)
STEPS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # UP, RIGHT, DOWN, LEFT


class PelletGuide:
    def __init__(self, initial_screen):
        # Pellets share the walls' colour. Their small connected components
        # must be removed from the wall mask before building the graph.
        coloured = np.all(initial_screen[:170] == WALL_COLOUR, axis=2).astype(np.uint8)
        _, components, stats, _ = cv2.connectedComponentsWithStats(coloured, connectivity=4)
        area = stats[components, cv2.CC_STAT_AREA]
        walls = (components != 0) & (area >= 32)

        points = {}
        pellets = {}
        for row in range(14):
            for column, (offset, bit) in enumerate(PELLET_BITS):
                x = 9 + column * 8 + (4 if column >= 9 else 0)
                y = 8 + row * 12
                if walls[y, x]:
                    continue
                node = (row, column)
                points[node] = (x, y)
                if components[y, x] and area[y, x] == 8:
                    pellets[node] = (59 + 3 * row + offset, 1 << bit)
        if not 100 <= len(pellets) <= 180:
            raise RuntimeError(f"Expected the original MsPacMan maze, found {len(pellets)} pellet sites.")

        graph = {node: [] for node in points}
        width = walls.shape[1]
        for (row, col), (x, y) in points.items():
            for direction, (dr, dc) in enumerate(STEPS):
                nc = col + dc
                other = row + dr, nc % len(PELLET_BITS)
                if other not in points:
                    continue
                nx, ny = points[other]
                if nc != other[1]:
                    first, last = sorted((x, nx))
                    clear = not (walls[y, :first + 1].any() or walls[y, last:].any())
                    distance = width - (last - first)
                else:
                    clear = not walls[min(y, ny):max(y, ny) + 1, min(x, nx):max(x, nx) + 1].any()
                    distance = abs(nx - x) + abs(ny - y)
                if clear:
                    graph[row, col].append((other, distance, direction))

        self.positions = np.asarray(list(points.values()), dtype=np.int16)
        self.routes = tuple(self._pellet_routes(start, graph, pellets) for start in points)

    @staticmethod
    def _pellet_routes(start, graph, pellets):
        best = {start: 0}
        frontier = [(0, NO_DIRECTION, start)]
        routes = []
        while frontier:
            distance, first, node = heappop(frontier)
            if distance != best[node]:
                continue
            if node in pellets:
                address, mask = pellets[node]
                routes.append((address, mask, first))
            for neighbour, step_cost, direction in graph[node]:
                new_distance = distance + step_cost
                if new_distance < best.get(neighbour, float("inf")):
                    best[neighbour] = new_distance
                    first_step = direction if first == NO_DIRECTION else first
                    heappush(frontier, (new_distance, first_step, neighbour))
        return tuple(routes)

    def direction(self, ram):
        # OCAtari exposes the sprite top left. Offset it to approximate its centre.
        player_x, player_y = int(ram[10]) - 9, int(ram[16]) + 6
        offsets = self.positions - (player_x, player_y)
        distance = np.abs(offsets).sum(axis=1)
        index = int(distance.argmin())
        if distance[index] > 16:
            return NO_DIRECTION
        for address, mask, direction in self.routes[index]:
            if not (int(ram[address]) & mask):
                continue
            if direction != NO_DIRECTION:
                return direction
            if distance[index] > 2:
                dx, dy = offsets[index]
                if abs(dx) > abs(dy):
                    return 1 if dx > 0 else 3
                return 2 if dy > 0 else 0
        return NO_DIRECTION
