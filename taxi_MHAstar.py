import numpy as np
from haversine import haversine
import operator
import heapq
import random
np.set_printoptions(linewidth=75)
import copy
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep
import gmplot
import os


class PriorityQueue:
	def __init__(self):
		self.elements = []
		self.set = set()

	def minkey(self):
		if not self.empty():
			return self.elements[0][0]
		else:
			return float('inf')
	
	def empty(self):
		return len(self.elements) == 0

	def put(self, item, priority):
		if item not in self.set:
			heapq.heappush(self.elements, (priority, item))
			self.set.add(item)
		else:
			# update
			# print("update", item)
			temp = []
			(pri, x) = heapq.heappop(self.elements)
			while x != item:
				temp.append((pri, x))
				(pri, x) = heapq.heappop(self.elements)
			temp.append((priority, item))
			for (pro, xxx) in temp:
				heapq.heappush(self.elements, (pro, xxx))

	def remove_element(self, item):
		if item in self.set:
			self.set.remove(item)
			temp = []
			(pro, x) = heapq.heappop(self.elements)
			while x != item:
				temp.append((pro, x))
				(pro, x) = heapq.heappop(self.elements)
			for (prito, yyy) in temp:
				heapq.heappush(self.elements, (prito, yyy))
	
	def top_show(self):
		return self.elements[0][1]
	
	def get(self):
		(priority, item) = heapq.heappop(self.elements)
		self.set.remove(item)
		return (priority, item)


def expand_state(s, j, visited, g_function, close_list_anchor, close_list_inad,  open_list, back_pointer, fgoal):
	for itera in range(n_heuristic):
		open_list[itera].remove_element(s)

	some_dic = {}
	for i in range(len(actual[s])):
		some_dic[i] = actual[s][i]

	sorted_dic = sorted(some_dic.items(), key=operator.itemgetter(1))
	neighbors = [sorted_dic[1][0], sorted_dic[2][0], sorted_dic[3][0]]
	
	for neigh in neighbors:
		if neigh not in visited:
			visited.add(neigh)
			back_pointer[neigh] = -1
			g_function[neigh] = float('inf')
		
		if g_function[neigh] > g_function[s] + actual[s][neigh]:
			g_function[neigh] = g_function[s] + actual[s][neigh]
			back_pointer[neigh] = s
			if neigh not in close_list_anchor:
				open_list[0].put(neigh, key(neigh, 0, fgoal, g_function, heuristics))
				if neigh not in close_list_inad:
					for var in range(1,n_heuristic):
						if key(neigh, var, fgoal, g_function, heuristics) <= W2 * key(neigh, 0, fgoal, g_function, heuristics):
							open_list[j].put(neigh, key(neigh, var, fgoal, g_function, heuristics))


def do_something(back_pointer, goal, start):
	l = [goal]
	x = back_pointer[goal]
	while x != start:
		l.append(x)
		x = back_pointer[x]
	l.append(x)
	l.reverse()
	print("path from " + str(start) + " to " + str(goal))
	print(l)
	return l
	quit()

class Nodes:
	def __init__(self, number, name, lat_, long_):
		self.number = number
		self.name = name
		self.lat = lat_
		self.long = long_
		
	def print_node(self):
		print(self.number, self.name, self.lat, self.long)
	
	def find_distance(self, other):
		p, q = self.lat, self.long
		x, y = other.lat, other.long
		d = haversine((p, q), (x, y))
		return d


def key(start, i, goal, g_function, heuristics):
	ans = g_function[start] + W1 * heuristics[i][start, goal]
	return ans

def MHA_star(nodes, heuristics, actual, start, goal, n_heuristic):
	
	g_function = {start.number: 0, goal.number: float('inf')}
	back_pointer = {start.number:-1, goal.number:-1}
	open_list = []
	visited = set()

	for i in range(n_heuristic):
		open_list.append(PriorityQueue())
		open_list[i].put(start.number, key(start.number, i, goal.number, g_function, heuristics))

	close_list_anchor = []
	close_list_inad = []

	queop = 0
	while open_list[0].minkey() < float('inf'):
		queop += 1
		for i in range(1, n_heuristic):
			if open_list[i].minkey() <= W2 * open_list[0].minkey():
				if g_function[goal.number] <= open_list[i].minkey():
					if g_function[goal.number] < float('inf'):
						print("queue operations: " + str(queop))
						return do_something(back_pointer, goal.number, start.number)
				else:
					_, get_s = open_list[i].top_show()
					visited.add(get_s)
					expand_state(get_s, i, visited, g_function, close_list_anchor, close_list_inad, open_list, back_pointer, goal.number)
					close_list_inad.append(get_s)
			else:
				if g_function[goal.number] <= open_list[0].minkey():
					if g_function[goal.number] < float('inf'):
						print("queue operations: " + str(queop))
						return do_something(back_pointer, goal.number, start.number)
				else:
					get_s = open_list[0].top_show()
					visited.add(get_s)
					expand_state(get_s, 0, visited, g_function, close_list_anchor, close_list_inad, open_list, back_pointer, goal.number)
					close_list_anchor.append(get_s)
	print("No path found to goal")
	return None


coordinates = [
('Rashtrapati Bhavan, Delhi', (28.6141527, 77.1959622)), 
('Bangla Sahib, Delhi', (28.6263529, 77.2090803)), 
('Metro Museum, Delhi', (28.6232969, 77.2144746)), 
('India Gate, Delhi', (28.612912, 77.2295097)), 
('Rajpath, Delhi', (28.6136152, 77.2150845)),
('Nehru Museum, Delhi', (28.6026029, 77.1987395)), 
('Delhi Haat, Delhi', (28.572745, 77.2090213)),
('Hauz Khas Village, Delhi', (28.5533997, 77.1941654)), 
('Qutab Minar, Delhi', (28.5244281, 77.1854559)), 
('Lotus Temple, Delhi', (28.553492, 77.25882639999999)), 
('Lajpat Nagar', (28.567593, 77.245519)),
('Nizamuddin Aulia Dargah', (28.591152, 77.241843)),
('IIITD, Delhi', (28.5456282, 77.2731505))
]


def route_display(path):
	url = "file://"+os.getcwd()+"/map.html"

	driver = webdriver.Chrome("/home/mohit/Desktop/chromedriver") #change location according to where you download the driver for selenium

	lat = []
	lon = []

	l = path

	for i in l:
	    lat.append(coordinates[i][1][0])
	    lon.append(coordinates[i][1][1])
	

	gmap = gmplot.GoogleMapPlotter(lat[0],lon[0],18)

	l_lat=[]
	l_lon=[]

	for x in coordinates:
	    l_lat.append(x[1][0])
	    l_lon.append(x[1][1])


	gmap.plot(l_lat,l_lon,'blue', edge_width=10)
	gmap.scatter(l_lat,l_lon,"red")

	llat = []

	llon = []

	for i in range(len(lon)):
	    gmap = gmplot.GoogleMapPlotter(lat[i],lon[i],13)	   
	    gmap.scatter(l_lat,l_lon,"red", edge_width=10)
	    gmap.scatter(l_lat[0:1:1],l_lon[0:1:1],"blue", edge_width=10)
	    gmap.scatter(l_lat[len(l_lat)-1:len(l_lat):1],l_lon[len(l_lat)-1:len(l_lat):1],"green", edge_width=10)
	   
	    llat.append(lat[i])
	    llon.append(lon[i])

	    gmap.plot(llat[0:len(llat)-1:1],llon[0:len(llon)-1:1],'blue', edge_width=5)
	    gmap.plot(llat[len(llat)-2:len(llat):1],llon[len(llat)-2:len(llat):1],'green', edge_width=5)

	    gmap.draw('map.html')
	    driver.get(url)
	    sleep(4)


if __name__ == '__main__':
	nodes = []
	n_heuristic = 2 # one consistent and other inconsistent
	W1 = 1.0
	W2 = 1.0
	actual = np.load('actual_distance.npy')
	heuristic_1 = np.load('actual_time.npy')

	for idx, l in enumerate(coordinates):
		nodes.append(Nodes(idx, l[0], l[1][0], l[1][1]))
	
	for node in nodes:
		node.print_node()
	
	consistent_heuristic = np.zeros((len(nodes), len(nodes)))
	for i in nodes:
		for j in nodes:
			if i.name != j.name:
				consistent_heuristic[i.number][j.number] = i.find_distance(j)
			else:
				consistent_heuristic[i.number][j.number] = 0

	heuristics = {1: heuristic_1, 0: consistent_heuristic}


	path = MHA_star(nodes, heuristics, actual, nodes[0], nodes[len(nodes)-1], n_heuristic)
	# path = [0,4,3,10,11,12]
	
	if path==None:
		print("None")
	else:
		route_display(path)