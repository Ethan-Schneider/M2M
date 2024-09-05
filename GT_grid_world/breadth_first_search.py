import numpy as np
import grid
import collections
import heapq

def breadth_first_search(graph : grid.Graph, start : tuple, goal : tuple):
    frontier = collections.deque()
    frontier.append(start)
    
    came_from = {}
    came_from[start] = None
    
    while frontier:
        current = frontier.popleft()
        print(current)
        
        if current == goal:
            break
        
        for next in graph.get_neighbors(current):
            if next not in came_from:
                print(next)
                frontier.append(next)
                came_from[next] = current
    return came_from


def draw_tile(graph, id, style):
    r = " . "
    if 'number' in style and id in style['number']: r = " %-2d" % style['number'][id]
    if 'point_to' in style and style['point_to'].get(id, None) is not None:
        (x1, y1) = id
        (x2, y2) = style['point_to'][id]
        if x2 == x1 + 1: r = " > "
        if x2 == x1 - 1: r = " < "
        if y2 == y1 + 1: r = " v "
        if y2 == y1 - 1: r = " ^ "
    if 'path' in style and id in style['path']:   r = " @ "
    if 'start' in style and id == style['start']: r = " A "
    if 'goal' in style and id == style['goal']:   r = " Z "
    if id in graph.obstacles: r = "###"
    return r

def draw_grid(graph, **style):
    print("___" * graph.width)
    for y in range(graph.height):
        for x in range(graph.width):
            print("%s" % draw_tile(graph, (x, y), style), end="")
        print()
    print("~~~" * graph.width)
    
def main():
    obstacles = [(16, 16), (16, 17), (16, 18), (16, 19), (17, 16), (18, 16), (18, 18), (19, 18)]
    g = grid.Graph(20, 20, 4, obstacles)
    start = (0, 0)
    goal = (19, 19)
    parents = breadth_first_search(g, start, goal)
    print(parents)
    draw_grid(g, point_to=parents, start=start, goal=goal)


if __name__=="__main__":
    main()
                
        