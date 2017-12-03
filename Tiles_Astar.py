import pygame, sys, random
from pygame.locals import *
import numpy as np
import time, Queue
from copy import deepcopy
import collections
from hashable import hashable
# np.random.seed(0)

main_n = 4

shuffle=[]
class Node:
    def __init__(self, board):
        self.board = board
    

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
    for _ in range(80):
        x, y, l = get_moves(board, n)
        ind = np.random.choice(range(len(l)), 1)[0]
        board = make_moves(deepcopy(board), (x, y), l[ind])
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

def make_manhatten(n):
    l = []
    for i in range(n):
        for j in range(n):
            l.append((i,j))
    return l

def cost(board, depth, number, all_list, goal):
    future = 0
    for (i, j) in all_list:
        occupant = board[i][j]
        if occupant != -1:
            x, y = np.where(goal == occupant)
            man_hat = abs(i - x[0]) + abs(j - y[0])
            future += man_hat

    return depth + future*3

def start(board, n, goal):
    all_list = make_manhatten(n)
    nodeNum = 1
    closed  = {}
    queue = Queue.PriorityQueue()
    orig = (cost(board,0,nodeNum, all_list, goal),nodeNum,board,0,None) # (cost, nodeNum, board_conf, pdepth, parent)
    queue.put(orig)
    closed[hashable(board)] = True
    count = 0
    solution = None
    print board
    while not queue.empty() and not solution:
        entry = queue.get()
        junk1, junk2, parent, pdepth, grandpa = entry
        if np.array_equal(parent, goal):
            solution = entry
            break
        print parent,
        print "cost: ",junk1, "nodeNum:", junk2, 
        print pdepth,
        print "------------------------"
        count += 1
        x, y, moves = get_moves(parent, n)
        for mov in moves:
            print mov, 
            child = make_moves(deepcopy(parent), (x, y), mov)
            print child,
            if closed.get(hashable(child)):
                continue
            closed[hashable(child)] = True
            print closed[hashable(child)],
            nodeNum += 1
            depth = pdepth + 1
            priority = cost(child, depth, nodeNum, all_list, goal)
            print priority
            new_entry = (priority, nodeNum, child, depth, entry)
            queue.put(new_entry)
            if np.array_equal(child, goal) : solution = new_entry
        print "child done"

    if solution:
        print count, "entries expanded. Queue still has " , queue.qsize()
        path = []
        while solution:
            path.append(solution[2])
            solution = solution[4]
        path.reverse()
        return path
    else:
        return []

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

def run():
    n = main_n
    goal = get_goal(n)
    board = random_button(n)
    startTime = time.time()
    path = start(board, n, goal)
    endTime = time.time() - startTime
    l_move=[]
    s_state = path[0]
    
    for c in range(1,len(path)):
        next_state = path[c]
        l_move.append(sq_move(s_state, next_state))
        # print
        s_state = next_state 
    return l_move
# print final_moves

final_moves = run()


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

def main():
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


def terminate():
    pygame.quit()
    sys.exit()


def checkForQuit():
    for event in pygame.event.get(QUIT): 
        terminate() 
    for event in pygame.event.get(KEYUP): 
        if event.key == K_ESCAPE:
            terminate() 
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
        slideAnimation(board, move, 'Generating new puzzle...', animationSpeed=int(TILESIZE / 2))
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
    main()



