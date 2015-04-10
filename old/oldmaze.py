#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys, curses, random, time, itertools, math

class Maze:
    mcount = 0
    vector = u'↑'
    signal = None
    yxdict = {u'←':(0,-1), u'↑':(-1,0), u'→':(0,1), u'↓':(1,0)}
    mDict = {u'←':(u'↓',u'↑'), u'↑':(u'←',u'→'), u'→':(u'↑',u'↓'), u'↓':(u'→',u'←')}

    def __init__(self, code):
        self.code = code
        codelines = code.splitlines();
        self.matrix = map(lambda l: map(lambda c: c, l), codelines)
        self.moves = [self.left, self.right, self.forward]
        for x,line in enumerate(codelines) :
            if line.count(u'S'):
                self.position = self.start = (x, line.index(u'S'))
                self.matrix[x][line.index(u'S')] = u' '
            elif line.count(u'G'):
                self.goal = (x, line.index(u'G'))

    def reset(self):
        self.position = self.start
        self.vector = u'↑'
        self.signal = None

    def left(self):
        self.mcount += 1
        self.vector = self.mDict[self.vector][0]
        self.signal = None

    def right(self):
        self.mcount += 1
        self.vector = self.mDict[self.vector][1]
        self.signal = None

    def forward(self):
        self.mcount += 1
        y,x = self.yxdict[self.vector]
        cy,cx = self.position
        front = self.matrix[cy + y][cx + x]
        if front == u'█':
            self.signal = (u'B', u'U')
            return(u'B', u'U')
        elif front == u'G':
            self.position = (cy + y, cx + x)
            self.signal = (u'G', u'G')
            return(u'G', u'G')
        else:
            self.position = (cy + y, cx + x)
            self.signal = None

def setup():
    qStatus = itertools.product(range(len(maze.matrix)),
                                range(len(maze.matrix[0])),
                                maze.yxdict.keys(),
                                [None,
                                 (u'B', u'U'), (u'G', u'G'),
                                 (u' ', u' '), (u' ', u'█'),
                                 (u'█', u'█'), (u'G', u'█')])
    qStatActionPair = itertools.product(qStatus, maze.moves)
    Q = dict([(s_a, qValue) for s_a in qStatActionPair])
    qAlpha = 0.1
    qGamma = 0.9
    qEpsilon = 0.1

def qReward (status, newStatus):
    signal = newStatus[3]
    if signal == None:
        if status[:2] == newStatus[:2]: return 0.0
        else: return 1.0
    elif signal == (u'B', u'U'): return -10.0
    elif signal == (u'G', u'G'): return 10000.0
    elif signal == (u' ', u' '): return 3.0
    elif signal == (u' ', u'█'): return 0.5
    elif signal == (u'█', u'█'): return 0.0
    elif signal == (u'G', u'█'): return 8000.0

def printMaze ():
    '''Print maze and robot'''
    maxy,maxx = stdscr.getmaxyx()
    pad.clear()
    pad.addstr(0, 0, maze.mazeStr.encode('utf-8'))
    pad.addstr(maze.robotPos[0],maze.robotPos[1],
                    maze.robotDirec.encode('utf-8'))
    pad.refresh(0, 0, 3, 0, maxy - 3, maxx - 1)

def printStatus (s):
    window.addstr(1, 0, s)
    window.clrtoeol()
    window.refresh()

def printCount ():
    window.addstr(2, 0, str(counts).strip('[]')[-(window.getmaxyx()[1]) + 1:])
    window.clrtoeol()
    window.refresh()

def addCount (n):
    counts.append(n)

def clearCount ():
    global counts
    counts = []

def directMode ():
    curses.noecho()
    curses.cbreak()
    window.keypad(1)
    while True:
        c = window.getch()
        if c == curses.KEY_LEFT: maze.turnLeft()
        elif c == curses.KEY_RIGHT: maze.turnRight()
        elif c == curses.KEY_UP:
            signal = maze.moveAhead()
            if (signal == None):
                printStatus('')
            else:
                signal1, signal2 = signal
                printStatus('Signal: (%s,%s)' % (signal1.encode('utf-8'),
                                                signal2.encode('utf-8')))
        elif c == curses.KEY_DOWN:
            obj1,obj2 = maze.lookAhead()
            printStatus('Signal: Ahead(%s,%s)' % (obj1.encode('utf-8'),
                                           obj2.encode('utf-8')))
        elif c == ord('r'): maze.randomlyReorient()
        elif c == ord('c'): break
        printMaze()
        window.move(0, 0)
    window.keypad(0)
    curses.echo()
    curses.nocbreak()

if __name__ == '__main__':
    stdscr = curses.initscr()
    mazeFile = open('maze.txt', 'r')
    mazeStr = unicode(mazeFile.read(), 'utf-8')
    mazeFile.close()
    maze = Maze(mazeStr)
    counts = []
    setup()
    window = curses.newwin(3,stdscr.getmaxyx()[1])
    pad = curses.newpad(100, 5000)
    padPos = (0, 0)
    printMaze()
    try:
        directMode()
    finally:
        curses.endwin()
