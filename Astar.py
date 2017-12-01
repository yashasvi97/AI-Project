import heapq
import copy

class PriorityQueue:
	def __init__(self):
		self.elements=[]
	def length(self):
		return len(self.elements)
	def empty(self):
		return len(self.elements) == 0
	def print_ele(self):
		for i in self.elements:
			print i
		print
	def check(self,a,cost):
		for i in self.elements:
			if a in i[1]:
				if cost>=i[0]:
					return 1
		return 0
	def put(self,item,priority):
		heapq.heappush(self.elements,(priority,item))
	def get(self):
		return heapq.heappop(self.elements)



grid_size = 20

grid = []

def init(grid_size):
	for j in range(grid_size):
		row = [0 for i in range(grid_size)]
		grid.append(row)

def grid_cord(a):
	return (grid_size-a[1]-1,a[0])

def blocks(obs):
	for i in obs:
		cord = grid_cord(i)
		# print i,cord
		grid[cord[0]][cord[1]]=1

def create_blocks(a,b):
	x1 = a[0]
	y1 = a[1]
	x2 = b[0]
	y2 = b[1]
	for i in range(x1,x2+1):
		for j in range(y1,y2+1):
			cord = grid_cord((i,j))
			grid[cord[0]][cord[1]]=1


def heuristic(a,b):
	# print a,b
	return abs(a[0]-b[0])+abs(a[1]-b[1])

# def heuristic(a,b):
# 	# print a,b
# 	return pow((pow((a[0]-b[0]),2)+pow(abs(a[1]-b[1]),2)),0.5)


def neighbor(a):
	neighbors=[]
	i=a[0]
	j=a[1]
	if j+1 < grid_size:
		if grid[i][j+1]!=1:
			neighbors.append((i,j+1))
	if j-1 >= 0:
		if grid[i][j-1]!=1:
			neighbors.append((i,j-1))
	if i+1 < grid_size:
		if grid[i+1][j]!=1:
			neighbors.append((i+1,j))
	if i-1 >= 0:
		if grid[i-1][j]!=1:
			neighbors.append((i-1,j))
	return neighbors
	# print neighbors


start = (0,0)
end = (grid_size-1,grid_size-1)


init(grid_size)



# create_blocks((1,1),(4,5))
# create_blocks((10,1),(18,14))
# create_blocks((15,17),(19,17))
# create_blocks((1,12),(3,16))
# create_blocks((1,16),(12,18))

# create_blocks((0,1),(19,1))

# print
# for i in grid:
# 	for j in i:
# 		if j==0:
# 			print "*",
# 		else:
# 			print "#",
# 	print


queue = PriorityQueue()
queue.put([grid_cord(start)],heuristic(start,end))
# print queue


ans = None

while queue.length()!=0:
	ele = queue.get()
	if ele[1][-1]==grid_cord(end):
		ans = ele
		break
	neighbors = neighbor(ele[-1][-1])
	
	for i in neighbors:
		if i not in ele[-1]:
			temp = copy.deepcopy(ele)
			current = temp[-1]
			cost = temp[0]
			new_cost = 1*(len(current)) + heuristic(i,grid_cord(end))

			if queue.check(i,new_cost)==0:
				current.append(i)
				queue.put(current,new_cost)

if ans==None:
	print "No Path found to goal"
else:
	for i in ans[1]:
		x=i[0]
		y=i[1]
		grid[x][y]="-"
	print
	print
	for i in grid:
		for j in i:
			if j==0:
				print "*",
			elif j==1:
				print "#",
			else:
				print j,
		print


# print
# for i in grid:
# 	for j in i:
# 		if j==0:
# 			print "*",
# 		else:
# 			print "#",
# 	print









