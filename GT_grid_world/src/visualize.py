from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

Colors = ['orange', 'blue', 'green']


class Animation:
  def __init__(self, map_size, obstacles, schedule):
    self.schedule = schedule
    self.combined_schedule = schedule
    
    self.fig = plt.figure(frameon=False, figsize=(map_size[0], map_size[1]))
    self.ax = self.fig.add_subplot(111, aspect='equal')
    self.fig.subplots_adjust(left=0,right=1,bottom=0,top=1, wspace=None, hspace=None)
    # self.ax.set_frame_on(False)

    self.patches = []
    self.artists = []
    self.agents = dict()
    self.agent_names = dict()
    # create boundary patch
    xmin = -0.5
    ymin = -0.5
    xmax = map_size[0] - 0.5
    ymax = map_size[1] - 0.5

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    self.patches.append(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor='none', edgecolor='red'))
    for o in obstacles:
      x, y = o[0], o[1]
      self.patches.append(Rectangle((y - 0.5, x - 0.5), 1, 1, facecolor='gray', edgecolor='gray'))

    # create agents:
    self.T = 0
    # draw goals first
    # for d, i in zip(map["agents"], range(0, len(map["agents"]))):
    #   self.patches.append(Rectangle((d["goal"][0] - 0.25, d["goal"][1] - 0.25), 0.5, 0.5, facecolor=Colors[0], edgecolor='black', alpha=0.5))
    
    for agent in schedule:
      name = agent
      self.agents[name] = Circle((schedule[agent][0]['x'], schedule[agent][0]['y']), 0.3, facecolor=Colors[0], edgecolor='black')
      self.agents[name].original_face_color = Colors[0]
      self.patches.append(self.agents[name])
      self.T = max(self.T, schedule[name][-1]['t'])
      self.agent_names[name] = self.ax.text(schedule[agent][0]['x'], schedule[agent][0]['y'], name.replace('agent', ''))
      self.agent_names[name].set_horizontalalignment('center')
      self.agent_names[name].set_verticalalignment('center')
      self.artists.append(self.agent_names[name])

    self.anim = animation.FuncAnimation(self.fig, self.animate_func,
                               init_func=self.init_func,
                               frames=int(self.T+1) * 10,
                               interval=100,
                               blit=True)

  def save(self, file_name, speed):
    self.anim.save(
      file_name,
      "ffmpeg",
      fps=10 * speed,
      dpi=200),
      # savefig_kwargs={"pad_inches": 0, "bbox_inches": "tight"})

  def show(self):
    plt.show()

  def init_func(self):
    for p in self.patches:
      self.ax.add_patch(p)
    for a in self.artists:
      self.ax.add_artist(a)
    return self.patches + self.artists

  def animate_func(self, i):
    for agent_name, agent in self.combined_schedule.items():
      pos = self.getState(i / 10, agent)
      p = (pos[1], pos[0])
      self.agents[agent_name].center = p
      self.agent_names[agent_name].set_position(p)

    # reset all colors
    for _,agent in self.agents.items():
      agent.set_facecolor(agent.original_face_color)

    # check drive-drive collisions
    agents_array = [agent for _,agent in self.agents.items()]
    for i in range(0, len(agents_array)):
      for j in range(i+1, len(agents_array)):
        d1 = agents_array[i]
        d2 = agents_array[j]
        pos1 = np.array(d1.center)
        pos2 = np.array(d2.center)
        if np.linalg.norm(pos1 - pos2) < 0.7:
          d1.set_facecolor('red')
          d2.set_facecolor('red')
          print("COLLISION! (agent-agent) ({}, {})".format(i, j))

    return self.patches + self.artists


  def getState(self, t, d):
    idx = 0
    while idx < len(d) and d[idx]["t"] < t:
      idx += 1
    if idx == 0:
      return np.array([float(d[0]["x"]), float(d[0]["y"])])
    elif idx < len(d):
      posLast = np.array([float(d[idx-1]["x"]), float(d[idx-1]["y"])])
      posNext = np.array([float(d[idx]["x"]), float(d[idx]["y"])])
    else:
      return np.array([float(d[-1]["x"]), float(d[-1]["y"])])
    dt = d[idx]["t"] - d[idx-1]["t"]
    t = (t - d[idx-1]["t"]) / dt
    pos = (posNext - posLast) * t + posLast
    return pos


def main(map_dimensions : tuple, obstacles : list, schedule : list, video : str = None, speed : int = 1):
  combined_schedule = {}
   
  for robot_num, sequence in enumerate(schedule):
    robot_path = []
    for i, step in enumerate(sequence):
      robot_path.append({'t':i, 'x':step[0], 'y':step[1]})
    combined_schedule['agent'+str(robot_num)] = robot_path
  
  animation = Animation(map_dimensions, obstacles, combined_schedule)
  
  
  if video:
    animation.save(video, speed)
  else:
    animation.show()
