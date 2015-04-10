# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')

import sys, curses, itertools, locale
locale.setlocale(locale.LC_ALL, 'ja_JP.UTF-8')
from brian import *
from brian.library.IF import *
import random, re, math
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl

defaultclock.dt = 1*ms
dt = defaultclock.dt

Vl = -70*mV
Vr = -70*mV
tau = 20*ms
tauP = 20*ms
beta = 0.9
gamma = 0.01
Ap = 1
Am = -1
lif_eqs = Equations('''
dVm/dt=(Vl-Vm)/tau : volt
K : 1
''')
# ------------------------------------------------------------------------------------------------------------------------------

inputs = 18
outputs = 4
OT = 1
NT = 180
HT = 180
Ns = 500
G = NT/inputs

NI = PoissonGroup(NT,rates=100*Hz)
S = []
E = []
I = []
for i in range(inputs):
    S.append(NI[G*i:G*(i+1)])
    E.append(NI[G*i:G*i+G/2])
    I.append(NI[G*i+G/2:G*(i+1)])

NH = NeuronGroup(HT, model=lif_eqs, threshold=-54*mV, reset=Vr)

NO = []
for i in range(outputs):
    NO.append(NeuronGroup(OT, model=lif_eqs, threshold=-54*mV, reset=Vr))

NI.Vm = Vr
NH.Vm = Vr

for i in range(outputs):
    NO[i].Vm = Vr

# ------------------------------------------------------------------------------------------------------------------------------

WIH = []
WEH = []
TIH = []
TEH = []
Pp_IH = []
Pp_EH = []
Pm_IH = []
Pm_EH = []
SM_Pp_IH = []
SM_Pm_IH = []
SM_Pp_EH = []
SM_Pm_EH = []

for i in range(inputs):
    WIH.append(Connection(I[i], NH, 'Vm', weight=rand(len(I[i]),len(NH))*-5*mV,structure='dense'))
    WEH.append(Connection(E[i], NH, 'Vm', weight=rand(len(E[i]),len(NH))*5*mV,structure='dense'))
    TIH.append(Connection(I[i], NH, 'K', weight= 0.0000001,structure='dense'))
    TEH.append(Connection(E[i], NH, 'K', weight= 0.0000001,structure='dense'))
    Pp_IH.append(NeuronGroup(G/2, 'dPp/dt=-Pp/tauP:1'))
    Pm_IH.append(NeuronGroup(HT, 'dPm/dt=-Pm/tauP:1'))
    Pp_EH.append(NeuronGroup(G/2, 'dPp/dt=-Pp/tauP:1'))
    Pm_EH.append(NeuronGroup(HT, 'dPm/dt=-Pm/tauP:1'))

WHO = []
THO = []
Pp_HO = []
Pm_HO = []

for i in range(outputs):
    WHO.append(Connection(NH[(HT/outputs)*i:(HT/outputs)*(i+1)], NO[i], 'Vm', weight=rand(HT/outputs,len(NO[i]))*5*mV,structure='dense'))
    THO.append(Connection(NH[(HT/outputs)*i:(HT/outputs)*(i+1)], NO[i], 'K', weight= 0.0000001,structure='dense'))
    Pp_HO.append(NeuronGroup(HT/outputs, 'dPp/dt=-Pp/tauP:1'))
    Pm_HO.append(NeuronGroup(OT, 'dPm/dt=-Pm/tauP:1'))

def pre_update_IH_builder(n):
    def pre_update_IH(spikes):
        if len(spikes):
            Pp_IH[n].Pp[spikes] += Ap
            for i in spikes:
                TIH[n].W[i, :] += Pm_IH[n].Pm
    pre_update_IH.__name__ = 'pre_update_IH'+str(n)
    return pre_update_IH

def post_update_IH_builder(n):
    def post_update_IH(spikes):
        if len(spikes):
            Pm_IH[n].Pm[spikes] += Am
            for i in spikes:
                TIH[n].W[:, i] += Pp_IH[n].Pp
    post_update_IH.__name__ = 'post_update_IH'+str(n)
    return post_update_IH

def pre_update_EH_builder(n):
    def pre_update_EH(spikes):
        if len(spikes):
            Pp_EH[n].Pp[spikes] += Ap
            for i in spikes:
                TEH[n].W[i, :] += Pm_EH[n].Pm
    pre_update_EH.__name__ = 'pre_update_EH'+str(n)
    return pre_update_EH

def post_update_EH_builder(n):
    def post_update_EH(spikes):
        if len(spikes):
            Pm_EH[n].Pm[spikes] += Am
            for i in spikes:
                TEH[n].W[:, i] += Pp_EH[n].Pp
    post_update_EH.__name__ = 'post_update_EH'+str(n)
    return post_update_EH

def pre_update_HO_builder(n):
    def pre_update_HO(spikes):
        if len(spikes):
            Pp_HO[n].Pp[spikes] += Ap
            for i in spikes:
                THO[n].W[i, :] += Pm_HO[n].Pm
    pre_update_HO.__name__ = 'pre_update_HO'+str(n)
    return pre_update_HO

def post_update_HO_builder(n):
    def post_update_HO(spikes):
        if len(spikes):
            Pm_HO[n].Pm[spikes] += Am
            for i in spikes:
                THO[n].W[:, i] += Pp_HO[n].Pp
    post_update_HO.__name__ = 'post_update_HO'+str(n)
    return post_update_HO

for i in range(inputs):
    SM_Pp_IH.append(SpikeMonitor(I[i], function=pre_update_IH_builder(i)))
    SM_Pm_IH.append(SpikeMonitor(NH, function=post_update_IH_builder(i)))
    SM_Pp_EH.append(SpikeMonitor(E[i], function=pre_update_EH_builder(i)))
    SM_Pm_EH.append(SpikeMonitor(NH, function=post_update_EH_builder(i)))

SM_Pp_HO = []
SM_Pm_HO = []

for i in range(outputs):
    SM_Pp_HO.append(SpikeMonitor(NH[(HT/outputs)*i:(HT/outputs)*(i+1)], function=pre_update_HO_builder(i)))
    SM_Pm_HO.append(SpikeMonitor(NO[i], function=post_update_HO_builder(i)))

@network_operation() 
def reduce_trace():
    for i in range(inputs):
        TIH[i].W *= beta
        TEH[i].W *= beta
    for i in range(outputs):
        THO[i].W *= beta

    rews = []

    for i in range(outputs):
        rews.append(reward[i])

    for i in range(inputs):
        clip(WIH[i].W + np.dot(TIH[i].W,np.diag(np.repeat(rews,HT/outputs)))*dt*gamma, -5.*mV, 0.*mV, WIH[i].W)
        clip(WEH[i].W + np.dot(TEH[i].W,np.diag(np.repeat(rews,HT/outputs)))*dt*gamma, 0.*mV, 5.*mV, WEH[i].W)
    for i in range(outputs):
        clip(WHO[i].W + THO[i].W*reward[i]*dt*gamma, 0*mV, 5*mV, WHO[i].W)

# ------------------------------------------------------------------------------------------------------------------------------

def policy_builder(n):
    def policy(spikes):
        if 0 in spikes:
            actions[n] += 1
            reward[n] = maze.rewardstore[(maze.state[0][0], maze.state[1][0]),maze.actions[n]]
            rsum[0] += reward[n]
        else:
            reward[n] = 0
    policy.__name__ = 'policy'+str(n)
    return policy

RM = []

for i in range(outputs):
    RM.append(SpikeMonitor(NO[i],function=policy_builder(i)))

# ------------------------------------------------------------------------------------------------------------------------------
class Maze:
    dyx = {'N':(-1,0), 'E':(0,1), 'W':(0,-1), 'S':(1,0)}

    def __init__(self, code):
        self.bot = u'▪'
        self.code = code
        codelines = code.splitlines();
        self.matrix = np.array(map(lambda l: map(lambda c: c, l), codelines))
        self.states = np.where((self.matrix == u' ') | (self.matrix == u'◈') | (self.matrix == u'S'))
        self.goal = np.where(self.matrix == u'◈')
        self.state = self.start = np.where(self.matrix == u'S')
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
    
    def reset(self):
        self.state = self.start

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

def printStatus(s):
    window.addstr(s)
    window.clrtoeol()
    window.refresh()

def printMaze ():
    maxy,maxx = stdscr.getmaxyx()
    pad.clear()
    pad.addstr(0, 0, maze.code.encode('utf-8'))
    pad.addstr(maze.state[0],maze.state[1],maze.bot.encode('utf-8'))
    pad.refresh(0, 0, 3, 0, maxy-3, maxx-1)
# ------------------------------------------------------------------------------------------------------------------------------

stdscr = curses.initscr()
mazeFile = open('maze.txt', 'r')
mazeStr = unicode(mazeFile.read(), 'utf-8')
mazeFile.close()
maze = Maze(mazeStr)
window = curses.newwin(3,stdscr.getmaxyx()[1])
pad = curses.newpad(100, 5000)
printMaze()

actions = [0]*outputs
reward = [0]*outputs
rsum = [0]
x = 0

try:
    while True:
        x += 1
        for ng in range(inputs):
            S[ng] = 10*Hz
        S[maze.state[0]] = 100*Hz
        S[inputs/2+maze.state[1]] = 100*Hz
        run(500*ms)
        inds = [a for a, val in enumerate(actions) if val == max(actions)]
        if maze.act(random.choice(inds)):
            maze.reset()
            printStatus("{0}-".format(x))
            x = 0
        printMaze()
        actions = [0]*outputs
        rsum = [0]
finally:
    curses.endwin()