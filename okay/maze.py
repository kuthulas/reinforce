#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys, curses, random, time, itertools, math, numpy, locale
locale.setlocale(locale.LC_ALL, 'ja_JP.UTF-8')

class Maze:
    dyx = {'N':(-1,0), 'E':(0,1), 'W':(0,-1), 'S':(1,0)}

    def __init__(self, code):
        self.bot = u'▪'
        self.code = code
        codelines = code.splitlines();
        self.matrix = numpy.array(map(lambda l: map(lambda c: c, l), codelines))
        self.states = numpy.where((self.matrix == u' ') | (self.matrix == u'◈') | (self.matrix == u'S'))
        self.goal = numpy.where(self.matrix == u'◈')
        self.state = self.start = numpy.where(self.matrix == u'S')
        self.actions = ['N','E','W','S']
        self.rewardstore()

    def rewardstore(self):
        self.rewardstore = dict.fromkeys(list(itertools.product(zip(self.states[0], self.states[1]), self.actions)), -1)
        self.rewardstore[(7,1),'E'] = 1
        self.rewardstore[(7,2),'E'] = 1
        self.rewardstore[(7,3),'E'] = 1
        self.rewardstore[(7,4),'E'] = 1
        self.rewardstore[(7,5),'N'] = 1
        self.rewardstore[(6,5),'N'] = 1
        self.rewardstore[(5,5),'E'] = 1
        self.rewardstore[(5,6),'E'] = 1
        self.rewardstore[(5,7),'N'] = 1
        self.rewardstore[(4,7),'N'] = 1
        self.rewardstore[(3,7),'W'] = 1
        self.rewardstore[(3,6),'W'] = 1
        self.rewardstore[(3,5),'N'] = 1
        self.rewardstore[(2,5),'N'] = 1
        self.rewardstore[(1,5),'E'] = 1
        self.rewardstore[(1,6),'E'] = 1
        self.rewardstore[(1,7),'E'] = 1

    def act(self, a_index):
        y,x = self.dyx[self.actions[a_index]]
        cy,cx = self.state
        front = self.matrix[cy + y, cx + x]
        if front == u'█':
            self.reward = -1
            return False
        else:
            self.reward = self.rewardstore[(self.state[0][0], self.state[1][0]), self.actions[a_index]]
            self.state = [cy + y, cx + x]
            if front == u'◈':
                return True
            else:
                return False

def interact():
    curses.noecho()
    curses.cbreak()
    window.keypad(1)

    while True:
        printMaze()
        cmd = window.getstr()
        if cmd == '': 
            break
        window.clear()
    window.keypad(0)
    curses.echo()
    curses.nocbreak()

def printStatus(s):
    window.addstr(1, 0, s)
    window.clrtoeol()
    window.refresh()

def printMaze ():
    maxy,maxx = stdscr.getmaxyx()
    pad.clear()
    pad.addstr(0, 0, maze.code.encode('utf-8'))
    pad.addstr(maze.state[0],maze.state[1],maze.bot.encode('utf-8'))
    pad.refresh(0, 0, 3, 0, maxy-3, maxx-1)

if __name__ == '__main__':
    #stdscr = curses.initscr()
    mazeFile = open('maze.txt', 'r')
    mazeStr = unicode(mazeFile.read(), 'utf-8')
    mazeFile.close()
    maze = Maze(mazeStr)
    print maze.state
    # window = curses.newwin(3,stdscr.getmaxyx()[1])
    # pad = curses.newpad(100, 5000)
    # printMaze()
    # try:
    #     interact()
    # finally:
    #     curses.endwin()
