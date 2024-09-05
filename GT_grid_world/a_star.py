import numpy as np
import grid
import queue

def a_star(graph : grid.Graph, start : tuple, goal : tuple):
    frontier = queue.PriorityQueue()
    frontier.put((start, 0))
    
    came_from = {}
    cost_so_far = {}
    
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current = frontier.get()
        
        if current[0] == goal:
            break
        
        for next in graph.get_neighbors(current[0]):
            new_cost = cost_so_far[current[0]] + graph.get_cost(current[0], next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal)
                frontier.put((next, priority))
                came_from[next] = current[0]
    return came_from, cost_so_far

def heuristic(loc, goal):
    return ((loc[0] - goal[0])**2 + (loc[1] - goal[1])**2)**0.5

def reconstruct_path(came_from: dict, start, goal) -> list:

    current = goal
    path: list = []
    if goal not in came_from: # no path was found
        return []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return path


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
    start = (5, 5)
    goal = (19, 19)
    came_from, cost_so_far = a_star(g, start, goal)
    
    draw_grid(g, point_to=came_from, start=start, goal=goal)
    print()
    draw_grid(g, path=reconstruct_path(came_from, start=start, goal=goal))


if __name__=="__main__":
    main()
                
        