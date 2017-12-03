import numpy as np
import time, heapq
import pygame, sys, random
from pygame.locals import *
from copy import deepcopy
import collections
# from hashable import hashable
import hashlib
# np.random.seed(0)
W1 = 1
W2 = 1
main_n = 3
shuffle=[]

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

	def put(self, item, priority, n):
		if item not in self.set:
			heapq.heappush(self.elements, (priority, item))
			self.set.add(item)
		else:
			temp = []
			(pri, x) = heapq.heappop(self.elements)
			while not np.array_equal(np.array(x).reshape(n,n), np.array(item).reshape(n,n)):
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
			while not np.array_equal(x, item):
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

def get_man_dist(board, all_list, goal):
	future = 0
	for (i, j) in all_list:
		occupant = board[i][j]
		if occupant != -1:
			x, y = np.where(goal == occupant)
			man_hat = abs(i - x[0]) + abs(j - y[0])
			future += man_hat
	return future

def get_lc_dist(board, all_list, goal):
	linear_conflict = 0
	for i in range(board.shape[0]):
		row = board[i]
		for k in range(row.shape[0]):
			for l in range(row.shape[0]):
				if k != l:
					tj = row[k]
					tk = row[l]
					if tj != -1 and tk != -1:
						x, y = np.where(goal == tj)
						p, q = np.where(goal == tk)
						if x[0] == i and p[0] == i:
							if (k > l) and (y[0] < q[0]):
								linear_conflict += 2
	return linear_conflict

def get_mis_tile_dist(board, all_list, goal):
	some = 0
	for i in range(board.shape[0]):
		for j in range(board.shape[0]):
			if board[i][j] != -1:
				if board[i][j] != goal[i][j]:
					some += 1

	return some
					
def key(board, i, goal, g_function, all_list):
	ans = g_function[tuple(board.flatten())] + W1 * hueristics[i](board, all_list, goal)
	return ans

def consistent_hueristic(board, all_list, goal):
	return get_man_dist(board, all_list, goal) + get_lc_dist(board, all_list, goal)

def hueristic_1(board, all_list, goal):
	return 1.0 * get_man_dist(board, all_list, goal) + 2.0 * get_lc_dist(board, all_list, goal) + 3.0 * get_mis_tile_dist(board, all_list, goal)

def hueristic_2(board, all_list, goal):
	return 4.0 * get_man_dist(board, all_list, goal) + 2.0 * get_lc_dist(board, all_list, goal) + 2.0 * get_mis_tile_dist(board, all_list, goal)

def hueristic_3(board, all_list, goal):
	return 2.0 * get_man_dist(board, all_list, goal) + 3.0 * get_lc_dist(board, all_list, goal) + 5.0 * get_mis_tile_dist(board, all_list, goal)

def hueristic_4(board, all_list, goal):
	return 3.0 * get_man_dist(board, all_list, goal) + 1.0 * get_lc_dist(board, all_list, goal) + 1.0 * get_mis_tile_dist(board, all_list, goal)
	
def printBoard(board, n):
	for i in range(n):
		for j in range(n):
			# print(i, j),
			print str(board[i][j]) + "\t",
		print " "

def initialize(n):
	board = np.empty((n, n), dtype=np.int16)
	for i in range(n):
		for j in range(n):
			board[i][j] = i*n + j
	return board + 1

def get_goal(n):
	goal = np.arange(n*n).reshape((n,n))
	goal = goal + 1
	goal[n-1][n-1] = -1
	return goal

def random_button(n):
	global shuffle
	board = initialize(n)
	x, y = np.where(board == n*n)
	board[x[0]][y[0]] = -1
	orig = board
	for _ in range(60):
		x, y, l = get_moves(board, n)
		ind = np.random.choice(range(len(l)), 1)[0]
		board = make_moves(deepcopy(board), (x, y), l[ind])
		# print sq_move(board,orig)
        shuffle.append(sq_move(board,orig))
        orig = board
	return board

def get_moves(board, n):
	x, y = np.where(board == -1)
	i = x[0]
	j = y[0]
	if i == 0:
		if j == 0:
			return i, j, [(i, j+1), (i+1, j)]
		elif j == n-1:
			return i, j, [(i, j-1), (i+1, j)]
		else:
			return i, j, [(i, j-1), (i+1, j), (i, j+1)]
	elif i == n-1:
		if j == 0:
			return i, j, [(i-1, j), (i, j+1)]
		elif j == n-1:
			return i, j, [(i, j-1), (i-1, j)]
		else:
			return i, j, [(i, j-1), (i-1, j), (i, j+1)]
	else:
		if j == 0:
			return i, j, [(i-1, j), (i+1, j), (i, j+1)]
		elif j == n-1:
			return i, j, [(i-1, j), (i+1, j), (i, j-1)]
		else:
			return i, j, [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]

def make_moves(board, old, new):
	temp = board[old[0]][old[1]]
	board[old[0]][old[1]] = board[new[0]][new[1]]
	board[new[0]][new[1]] = temp
	return board

def sq_move(a, b):
	x, y = np.where(a == -1)
	l, m = np.where(b == -1)
	if (x[0] - l[0]) > 0:
		return 'up'
	elif (x[0] - l[0]) < 0:
		return 'down'
	else:
		if (y[0] - m[0]) > 0:
			return 'left'
		elif (y[0] - m[0]) < 0:
			return 'right'

def make_manhatten(n):
	l = []
	for i in range(n):
		for j in range(n):
			l.append((i,j))
	return l

def terminate(back_pointer, goal, board, n):
	path = [goal]
	hass_start = tuple(board.flatten())
	x = back_pointer[tuple(goal.flatten())]
	while x != hass_start:
		path.append(np.array(x).reshape(n,n))
		x = back_pointer[x]
	path.append(board)
	path.reverse()
	orig = path[0]
	print orig,
	moves = []
	for c in range(1, len(path)):
		print sq_move(orig, path[c])
		moves.append(sq_move(orig, path[c]))
		print path[c],
		orig = path[c]
	# print moves
	return moves

def expand_state(s, j, visited, g_function, close_list_anchor, close_list_inad,  open_list, back_pointer, n_hueristic, n, all_list):

	for itera in range(n_hueristic):
		open_list[itera].remove_element(s)
	mat = np.array(s).reshape(n,n)
	x, y, moves = get_moves(mat, n)
	for mov in moves:
		child = make_moves(deepcopy(mat), (x, y), mov)
		child_hash = tuple(child.flatten())
		if child_hash not in visited:
			visited.add(child_hash)
			back_pointer[child_hash] = None
			g_function[child_hash] = float('inf')

		if g_function[child_hash] > g_function[s] + 1:
			g_function[child_hash] = g_function[s] + 1
			back_pointer[child_hash] = s
			if child_hash not in close_list_anchor:
				open_list[0].put(child_hash, key(child, 0, goal, g_function, all_list), n)
				if child_hash not in close_list_inad:
					for var in range(1,n_hueristic):
						if key(child, var, goal, g_function, all_list) <= W2 * key(child, 0, goal, g_function, all_list):
							open_list[j].put(child_hash, key(child, var, goal, g_function, all_list), n)


def mystart(board, n, goal, hueristics, n_hueristic):
	all_list = make_manhatten(n)
	start_hash = tuple(board.flatten())
	goal_hash = tuple(goal.flatten())

	g_function = {start_hash: 0, goal_hash: float('inf')}
	back_pointer = {start_hash: None, goal_hash: None}
	open_list = []
	visited = set()

	for i in range(n_hueristic):
		open_list.append(PriorityQueue())
		open_list[i].put(start_hash, key(board, i, goal, g_function, all_list),n)
	
	close_list_anchor = []
	close_list_inad = []
	startTime = time.time()
	globalTime = time.time()
	some_boring_var = 0
	total_time = 0
	while open_list[0].minkey() < float('inf'):
		some_boring_var += 1
		
		for i in range(1, n_hueristic):
			if open_list[i].minkey() <= W2 * open_list[0].minkey():
				
				if g_function[goal_hash] <= open_list[i].minkey():
					if g_function[goal_hash] < float('inf'):									
						return terminate(back_pointer, goal, board, n)						
				else:
					get_s = open_list[i].top_show()					
					expand_state(get_s, i, visited, g_function, close_list_anchor, close_list_inad, open_list, back_pointer, n_hueristic, n, all_list)
					close_list_inad.append(get_s)
			else:
				
				if g_function[goal_hash] <= open_list[0].minkey():					
					if g_function[goal_hash] < float('inf'):					
						
						return terminate(back_pointer, goal, board, n)
						
				else:					
					get_s = open_list[0].top_show()			
					expand_state(get_s, 0, visited, g_function, close_list_anchor, close_list_inad, open_list, back_pointer, n_hueristic, n, all_list)
					close_list_anchor.append(get_s)




BOARDWIDTH = main_n
BOARDHEIGHT = main_n 
TILESIZE = 100
WINDOWWIDTH = TILESIZE*BOARDWIDTH + 200
WINDOWHEIGHT = TILESIZE*BOARDWIDTH + 200
FPS = 30
BLANK = None
BLACK =         (  0,   0,   0)
WHITE =         (255, 255, 255)
BRIGHTBLUE =    (  0,  50, 255)
DARKTURQUOISE = (  3,  54,  73)
GREEN =         (  0, 204,   0)

BGCOLOR = BLACK
TILECOLOR = DARKTURQUOISE
TEXTCOLOR = WHITE
BORDERCOLOR = BRIGHTBLUE
BASICFONTSIZE = 20

BUTTONCOLOR = WHITE
BUTTONTEXTCOLOR = BLACK
MESSAGECOLOR = WHITE

XMARGIN = int((WINDOWWIDTH - (TILESIZE * BOARDWIDTH + (BOARDWIDTH - 1))) / 2)
YMARGIN = int((WINDOWHEIGHT - (TILESIZE * BOARDHEIGHT + (BOARDHEIGHT - 1))) / 2)

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'



final_moves = []


n = main_n
n_hueristic = 3
goal = get_goal(n)
board_algo = random_button(n)
startTime = time.time()
hueristics = {0: consistent_hueristic, 1: hueristic_1, 2: hueristic_2}
path = mystart(board_algo, n, goal, hueristics, n_hueristic)
final_moves = path

shuffle = deepcopy(final_moves)
shuffle.reverse()

def run_tiles():
    global FPSCLOCK, DISPLAYSURF, BASICFONT, RESET_SURF, RESET_RECT,SOLVE_SURF, SOLVE_RECT

    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
    pygame.display.set_caption('Tile Sliding')
    BASICFONT = pygame.font.Font('freesansbold.ttf', BASICFONTSIZE)
    RESET_SURF, RESET_RECT = makeText('',    TEXTCOLOR, TILECOLOR, WINDOWWIDTH - 120, WINDOWHEIGHT - 60)
    SOLVE_SURF, SOLVE_RECT = makeText('Solve',    TEXTCOLOR, TILECOLOR, WINDOWWIDTH - 120, WINDOWHEIGHT - 30)

    mainBoard = generateNewPuzzle()

    SOLVEDBOARD = getStartingBoard()
    allMoves = []

    while True:
        slideTo = None
        msg = '' 
        if mainBoard == SOLVEDBOARD:
            msg = 'Solved!'

        drawBoard(mainBoard, msg)

        checkForQuit()
        for event in pygame.event.get():
            if event.type == MOUSEBUTTONUP:
                if RESET_RECT.collidepoint(event.pos):
                    resetAnimation(mainBoard, allMoves)
                    allMoves = []
                elif SOLVE_RECT.collidepoint(event.pos):
                    resetAnimation(mainBoard)
                    allMoves = []

        if slideTo:
            slideAnimation(mainBoard, slideTo, '', 8) 
            makeMove(mainBoard, slideTo)
            allMoves.append(slideTo) 
        pygame.display.update()
        FPSCLOCK.tick(FPS)


def go_out():
    pygame.quit()
    sys.exit()


def checkForQuit():
    for event in pygame.event.get(QUIT): 
        go_out() 
    for event in pygame.event.get(KEYUP): 
        if event.key == K_ESCAPE:
            go_out() 
        pygame.event.post(event) 


def getStartingBoard():         
    counter = 1
    board = []
    for x in range(BOARDWIDTH):
        column = []
        for y in range(BOARDHEIGHT):
            column.append(counter)
            counter += BOARDWIDTH
        board.append(column)
        counter -= BOARDWIDTH * (BOARDHEIGHT - 1) + BOARDWIDTH - 1

    board[BOARDWIDTH-1][BOARDHEIGHT-1] = BLANK
    return board


def getBlankPosition(board):
    
    for x in range(BOARDWIDTH):
        for y in range(BOARDHEIGHT):
            if board[x][y] == BLANK:
                return (x, y)


def makeMove(board, move):
    
    blankx, blanky = getBlankPosition(board)

    if move == UP:
        board[blankx][blanky], board[blankx][blanky + 1] = board[blankx][blanky + 1], board[blankx][blanky]
    elif move == DOWN:
        board[blankx][blanky], board[blankx][blanky - 1] = board[blankx][blanky - 1], board[blankx][blanky]
    elif move == LEFT:
        board[blankx][blanky], board[blankx + 1][blanky] = board[blankx + 1][blanky], board[blankx][blanky]
    elif move == RIGHT:
        board[blankx][blanky], board[blankx - 1][blanky] = board[blankx - 1][blanky], board[blankx][blanky]


def getLeftTopOfTile(tileX, tileY):
    left = XMARGIN + (tileX * TILESIZE) + (tileX - 1)
    top = YMARGIN + (tileY * TILESIZE) + (tileY - 1)
    return (left, top)


def drawTile(tilex, tiley, number, adjx=0, adjy=0):
    left, top = getLeftTopOfTile(tilex, tiley)
    pygame.draw.rect(DISPLAYSURF, TILECOLOR, (left + adjx, top + adjy, TILESIZE, TILESIZE))
    textSurf = BASICFONT.render(str(number), True, TEXTCOLOR)
    textRect = textSurf.get_rect()
    textRect.center = left + int(TILESIZE / 2) + adjx, top + int(TILESIZE / 2) + adjy
    DISPLAYSURF.blit(textSurf, textRect)


def makeText(text, color, bgcolor, top, left):
    textSurf = BASICFONT.render(text, True, color, bgcolor)
    textRect = textSurf.get_rect()
    textRect.topleft = (top, left)
    return (textSurf, textRect)


def drawBoard(board, message):
    DISPLAYSURF.fill(BGCOLOR)
    if message:
        textSurf, textRect = makeText(message, MESSAGECOLOR, BGCOLOR, 5, 5)
        DISPLAYSURF.blit(textSurf, textRect)

    for tilex in range(len(board)):
        for tiley in range(len(board[0])):
            if board[tilex][tiley]:
                drawTile(tilex, tiley, board[tilex][tiley])

    left, top = getLeftTopOfTile(0, 0)
    width = BOARDWIDTH * TILESIZE
    height = BOARDHEIGHT * TILESIZE
    pygame.draw.rect(DISPLAYSURF, BORDERCOLOR, (left - 5, top - 5, width + 11, height + 11), 4)

    DISPLAYSURF.blit(RESET_SURF, RESET_RECT)
    DISPLAYSURF.blit(SOLVE_SURF, SOLVE_RECT)


def slideAnimation(board, direction, message, animationSpeed):

    blankx, blanky = getBlankPosition(board)
    if direction == UP:
        movex = blankx
        movey = blanky + 1
    elif direction == DOWN:
        movex = blankx
        movey = blanky - 1
    elif direction == LEFT:
        movex = blankx + 1
        movey = blanky
    elif direction == RIGHT:
        movex = blankx - 1
        movey = blanky

    drawBoard(board, message)
    baseSurf = DISPLAYSURF.copy()
    moveLeft, moveTop = getLeftTopOfTile(movex, movey)
    pygame.draw.rect(baseSurf, BGCOLOR, (moveLeft, moveTop, TILESIZE, TILESIZE))

    for i in range(0, TILESIZE, animationSpeed):
        checkForQuit()
        DISPLAYSURF.blit(baseSurf, (0, 0))
        if direction == UP:
            drawTile(movex, movey, board[movex][movey], 0, -i)
        if direction == DOWN:
            drawTile(movex, movey, board[movex][movey], 0, i)
        if direction == LEFT:
            drawTile(movex, movey, board[movex][movey], -i, 0)
        if direction == RIGHT:
            drawTile(movex, movey, board[movex][movey], i, 0)
        pygame.display.update()
        FPSCLOCK.tick(FPS)        


def generateNewPuzzle():
    moov = shuffle[0:len(shuffle)]
    board = getStartingBoard()
    drawBoard(board, '')
    pygame.display.update()
    pygame.time.wait(500)
    lastMove = None
    for i in range(len(moov)):
        move = moov[i]
        slideAnimation(board, move, 'Generating new puzzle...', animationSpeed=int(TILESIZE/1.5))
        makeMove(board, move)
        lastMove = move
    return board


def resetAnimation(board):
    revAllMoves = final_moves
    for move in revAllMoves:
        if move == UP:
            oppositeMove = DOWN
        elif move == DOWN:
            oppositeMove = UP
        elif move == RIGHT:
            oppositeMove = LEFT
        elif move == LEFT:
            oppositeMove = RIGHT
        slideAnimation(board, oppositeMove, '', animationSpeed=int(TILESIZE / 3))
        makeMove(board, oppositeMove)

if __name__ == '__main__':
	run_tiles()

