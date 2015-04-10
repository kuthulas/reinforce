import warnings
warnings.filterwarnings('ignore')

from brian import *
from brian.library.IF import *
import random, re, math
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
from collections import defaultdict

mpl.rcParams['toolbar'] = 'None'

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

inputs = 16
outputs = 4
OT = 1
NT = 160
HT = 160
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
tag = {"bd": 0, "bu": 1, "fu": 2, "fd": 3}
lengths = [0.4,0.3,0.1]

factor = math.pi/180

angles = [[factor*-120,factor*-20,factor*90], # back down
          [factor*-130,factor*-50,factor*90], # back up
          [factor*-30,factor*-90,factor*90], # front up
          [factor*-60,factor*-20,factor*90]] # front down

xlook = [[tag["bd"],tag["bd"]],
         [tag["bd"],tag["bu"]],
         [tag["bd"],tag["fu"]],
         [tag["bd"],tag["fd"]],
         [tag["bu"],tag["bd"]],
         [tag["bu"],tag["bu"]],
         [tag["bu"],tag["fu"]],
         [tag["bu"],tag["fd"]],
         [tag["fu"],tag["bd"]],
         [tag["fu"],tag["bu"]],
         [tag["fu"],tag["fu"]],
         [tag["fu"],tag["fd"]],
         [tag["fd"],tag["bd"]],
         [tag["fd"],tag["bu"]],
         [tag["fd"],tag["fu"]],
         [tag["fd"],tag["fd"]]]

def get_vector(angles,lengths):
    x = [0,0,0,0]
    y = [0,0,0,0]
    x[1] = lengths[0]*math.cos(angles[0])
    y[1] = lengths[0]*math.sin(angles[0])
    x[2] = x[1] + lengths[1]*math.cos(angles[0]+angles[1])
    y[2] = y[1] + lengths[1]*math.sin(angles[0]+angles[1])
    x[3] = x[2] + lengths[2]*math.cos(angles[0]+angles[1]+angles[2])
    y[3] = y[2] + lengths[2]*math.sin(angles[0]+angles[1]+angles[2])
    return x, y

def show_state(state):
    X1, Y1 = get_vector(angles[xlook[state][0]], lengths)
    X2, Y2 = get_vector(angles[xlook[state][1]], lengths)

    line1.set_data(X1,Y1)
    line2.set_data([x - 0.05 for x in X2],Y2)
    fig.canvas.draw()
# ------------------------------------------------------------------------------------------------------------------------------

xmove = np.matrix([
        [ 2,   4,   5,  13],
        [ 1,   3,   6,  14],
        [ 4,   2,   7,  15],
        [ 3,   1,   8,  16],
        [ 6,   8,   1,   9],
        [ 5,   7,   2,  10],
        [ 8,   6,   3,  11],
        [ 7,   5,   4,  12],
        [10,  12,  13,   5], 
        [ 9,  11,  14,   6],
        [12,  10,  15,   7],
        [11,   9,  16,   8],
        [14,  16,   9,   1],
        [13,  15,  10,   2],
        [16,  14,  11,   3],
        [15,  13,  12,   4]
    ], dtype="int" ) - 1

rmove = np.matrix([
        [ 1,  -1,   1,  -1],
        [-1,   1,  -1,  -1],
        [ 1,  -1,  -1,  -1],
        [-1,  -1,   1,  -1],
        [-1,  -1,  -1,   1],
        [ 1,  -1,   1,  -1],
        [ 1,  -1,   1,  -1],
        [-1,   2,  -1,   1],
        [-1,  -1,   1,  -1],
        [ 1,  -1,   1,  -1],
        [ 1,  -1,   1,  -1],
        [-1,   2,   1,  -1],
        [ 1,  -1,  -1,  -1],
        [-1,  -1,  -1,   2],
        [ 1,  -1,  -1,   2],
        [-1,  -1,   1,  -1]
    ], dtype="int")

def policy_builder(n):
    def policy(spikes):
        if 0 in spikes:
            actions[n] += 1
            reward[n] = rmove[state,n]
            rsum[0] += reward[n]
        else:
            reward[n] = 0
    policy.__name__ = 'policy'+str(n)
    return policy

RM = []

for i in range(outputs):
    RM.append(SpikeMonitor(NO[i],function=policy_builder(i)))

reward = [0]*outputs

state = random.randint(0,inputs-1)
actions = [0]*outputs
rsum = [0]

fig = pl.figure("Walk", figsize=(3, 3), dpi=80)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-1, 1), ylim=(-1, 1.5))
plt.tick_params(axis='both',which='both',bottom='off',top='off',left='off',right='off',labelbottom='off',labeltop='off', labelleft='off', labelright='off')
ax.grid()

line0, = ax.plot([-0.02,-0.02],[0.,.6], 'o-', lw=5, color='b')
line01, = ax.plot([-0.02,0.4],[0.6,.2], 'o-', lw=5, color='b')
line02, = ax.plot([-0.02,-0.4],[0.6,.2], 'o-', lw=5, color='b')

circle0 = pl.Circle((0,0.8),0.2, color='r')
fig.gca().add_artist(circle0)

line1, = ax.plot([], [], 'o-', lw=5)
line2, = ax.plot([], [], 'o-', lw=5)
line3, = ax.plot([], [], 'o-', lw=5)
line4, = ax.plot([], [], 'o-', lw=5)

show_state(state)
fig.show()

crewards = defaultdict(lambda  : defaultdict(list))

for x in range(150):
    for ng in range(inputs):
        if ng == state:
            S[ng].rate = 100*Hz
        else:
            S[ng].rate = 0*Hz
    run(500*ms)
    indices = [a for a, val in enumerate(actions) if val > 20]
    if not indices:
        inds = [a for a, val in enumerate(actions) if val == max(actions)]
        action = random.choice(inds)
    else:
        action = random.choice(indices)
    print("{0:>4}| State {1:>4}| Action {2:>4}| Reward {3:>6}".format(x+1, state+1, action+1, rsum[0]))
    state = xmove[state,action]
    show_state(state)
    actions = [0]*outputs
    crewards[state][actions.index(max(actions))].append(rsum[0])
    rsum = [0]

# fig2 = pl.figure()
# ax1 = fig.add_subplot(212)
# ax1.plot(crewards[0][1])
# ax2 = fig.add_subplot(211)
# ax2.plot(crewards[14][0])
# fig2.show()

gamma = 0

for x in range(50):
    for ng in range(inputs):
        if ng == state:
            S[ng].rate = 100*Hz
        else:
            S[ng].rate = 0*Hz
    run(500*ms)
    indices = [a for a, val in enumerate(actions) if val > 20]
    if not indices:
        inds = [a for a, val in enumerate(actions) if val == max(actions)]
        action = random.choice(inds)
    else:
        action = random.choice(indices)
    print("{0:>4}| State {1:>4}| Action {2:>4}".format(x+1, state+1, action+1))
    state = xmove[state,action]
    show_state(state)
    actions = [0]*outputs
