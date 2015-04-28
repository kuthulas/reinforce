import warnings
warnings.filterwarnings('ignore')

from brian import *
from brian.library.IF import *
import random, re, math, itertools, sys
import numpy as np

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
#[3,2,0]
inputs1 = 12 # [0,0,100,0],[0,100,0,0],[0,0,0,0]
inputs2 = 16 # [0,0,0] [0,0,0] ... 8 times
inputs = inputs1 + inputs2 
outputs = 16
OT = 1
NT = 560
HT = 320
Ns = 500
G = NT/inputs

NI = PoissonGroup(NT,rates=40*Hz)
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
    rsum[0] += sum(reward)
    for i in range(inputs):
        clip(WIH[i].W + np.dot(TIH[i].W,np.diag(np.repeat(rews,HT/outputs)))*dt*gamma, -5.*mV, 0.*mV, WIH[i].W)
        clip(WEH[i].W + np.dot(TEH[i].W,np.diag(np.repeat(rews,HT/outputs)))*dt*gamma, 0.*mV, 5.*mV, WEH[i].W)
    for i in range(outputs):
        clip(WHO[i].W + THO[i].W*reward[i]*dt*gamma, 0*mV, 5*mV, WHO[i].W)

# ------------------------------------------------------------------------------------------------------------------------------
# [0,1,2][3,4,5][6,7,8][9,10,11][12,13,14][15,16,17][18,19,20][21,22,23]

# i state -> run(500ms) -> 8 actions -> o state

def policy_builder(n):
    def policy(spikes):
        if len(spikes):
            AB[n] += len(spikes)
    policy.__name__ = 'policy'+str(n)
    return policy

RM = []

for i in range(outputs):
    RM.append(SpikeMonitor(NO[i],function=policy_builder(i)))

def update_state(cstate):
    for t in np.nonzero(cstate)[0]:
        if cstate[t]!=0:
            move = random.randint(0,1)
            if move == 0: #left
                if t > 0 and cstate[t-1] == 0:
                    cstate[t-1] = cstate[t]
                    cstate[t] = 0
                elif t < len(cstate)-1 and cstate[t+1] == 0:
                    cstate[t+1] = cstate[t]
                    cstate[t] = 0
            else:
                if t < len(cstate)-1 and cstate[t+1] == 0:
                    cstate[t+1] = cstate[t]
                    cstate[t] = 0
                elif t > 0 and cstate[t-1] == 0:
                    cstate[t-1] = cstate[t]
                    cstate[t] = 0
    return cstate

reward = [0]*outputs
neigh = np.array([[0,1,2,3],[2,3,4,5],[4,5,6,7]])
AB = np.array([0]*outputs)
actions = np.zeros(outputs/2)

while True:
    states = np.array([[3,3,0],[3,0,3],[0,3,3]])
    state = states[random.randint(0,2)]
    count = 0
    while True:
        for i in range(inputs):
            S[i].rate = 0*Hz
        for j in range(len(state)):
            S[(inputs1/len(state))*j+state[j]-1].rate = 40*Hz

        for q in range(outputs/2):
            S[int(2*q+actions[q])].rate = 40*Hz

        rsum = [0]
        AB = np.array([0]*outputs)
        print state,
        sys.stdout.flush()
        count += 1

        run(500*ms)
        actions = np.zeros(outputs/2)
        for q in range(outputs/2):
            BA = AB[(outputs/8)*q:(outputs/8)*q+2]
            actions[q] = np.argmax(BA)
            reward[q] = -1

        for m in range(len(state)):
            if state[m] != 0:
                neighbors = neigh[m]
                left = np.where(actions[neighbors[0:2]] == 1)
                ncleft = len(left[0])
                right = np.where(actions[neighbors[2:4]] == 0)
                ncright = len(right[0])

                for lt in left[0]:
                    reward[neighbors[lt]] = 2
                for rt in right[0]:
                    reward[neighbors[2+rt]] = 2

                if(ncleft+ncright > 2):
                    for lt in left[0]:
                        reward[neighbors[lt]] = 5
                    for rt in right[0]:
                        reward[neighbors[2+rt]] = 5
                    state[m] += -1
        print actions, state, AB
        state = update_state(state)
        if len(np.nonzero(state)[0]) == 0:
            print count
            break
