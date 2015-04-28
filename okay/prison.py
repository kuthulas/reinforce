import warnings
warnings.filterwarnings('ignore')

from brian import *
from brian.library.IF import *
import random, re, math
import numpy as np

defaultclock.dt = 1*ms
dt = defaultclock.dt

Vl = -70*mV
Vr = -70*mV
tau = 20*ms
tauP = 20*ms
beta = 0.9
gamma = 0.001
Ap = 1
Am = -1
lif_eqs = Equations('''
dVm/dt=(Vl-Vm)/tau : volt
K : 1
''')

# ------------------------------------------------------------------------------------------------------------------------------

inputs = 4
outputs = 4
OT = 1
NT = 240
HT = 240
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

    for i in range(inputs):
        clip(WIH[i].W + np.dot(TIH[i].W,np.diag(np.repeat(rews,HT/outputs)))*dt*gamma, -5.*mV, 0.*mV, WIH[i].W)
        clip(WEH[i].W + np.dot(TEH[i].W,np.diag(np.repeat(rews,HT/outputs)))*dt*gamma, 0.*mV, 5.*mV, WEH[i].W)
    for i in range(outputs):
        clip(WHO[i].W + THO[i].W*reward[i]*dt*gamma, 0*mV, 5*mV, WHO[i].W)

# ------------------------------------------------------------------------------------------------------------------------------
def policy_builder(n):
    def policy(spikes):
        if len(spikes):
            AB[n] += len(spikes)
    policy.__name__ = 'policy'+str(n)
    return policy

def set_rewards(array):
    CD1 = AB[0:inputs/2].index(max(AB[0:inputs/2]))
    CD2 = AB[(inputs/2):inputs].index(max(AB[(inputs/2):inputs]))
    reward[CD1] = rtable[CD1,CD2,0]
    reward[(inputs/2)+CD2] = rtable[CD1,CD2,1]
    reward[1-CD1] = math.copysign(0, -rtable[CD1,CD2,0])
    reward[(inputs/2)+(1-CD2)] = math.copysign(0, -rtable[CD1,CD2,1])
RM = []

for i in range(outputs):
    RM.append(SpikeMonitor(NO[i],function=policy_builder(i)))

AB = [0]*outputs
reward = [0]*outputs
crewards = []
rewards = []
perf = []
rsum = 0
rwa = 0
rwb = 0
rwsa = []
rwsb = []
rtable = np.array([[[4,4],[-3,5]],[[5,-3],[-2,-2]]])

for i in range(100):
    S[0].rate = AB[0:inputs/2].index(max(AB[0:inputs/2]))*40*Hz
    S[1].rate = (1-AB[0:inputs/2].index(max(AB[0:inputs/2])))*40*Hz
    S[2].rate = AB[(inputs/2):inputs].index(max(AB[(inputs/2):inputs]))*40*Hz
    S[3].rate = (1 - AB[(inputs/2):inputs].index(max(AB[(inputs/2):inputs])))*40*Hz
    AB = [0]*outputs
    run(500*ms)
    print AB,
    set_rewards(AB)
    print reward
    rsum += sum(reward)
    rwsa.append(AB[0]*reward[0] + AB[1]*reward[1])
    rwsb.append(AB[2]*reward[2] + AB[3]*reward[3])
    crewards.append(rsum)
    rewards.append(sum(reward))
    perf.append((AB[0]-AB[1])/40.)
print perf
plot(perf)
show()
print rewards
plot(rewards)
show()
print crewards
plot(crewards)
show()
print rwsa
plot(rwsa)
show()
print rwsb
plot(rwsb)
show()
