import warnings
warnings.filterwarnings('ignore')

from brian import *
from brian.library.IF import *
import random, re
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

a=0.02/ms
b=0.2/ms

eqs='''
dvm/dt=(0.04/ms/mV)*vm**2+(5/ms)*vm+140*mV/ms-w : volt
dw/dt=a*(b*vm-w)                            : volt/second
K : 1
'''

threshold = -50*mV
reset = AdaptiveReset(Vr=-65*mV, b=2.0*nA) 
# ------------------------------------------------------------------------------------------------------------------------------

inputs = 4
OT = 1
NT = 80
HT = 80
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
NO1 = NeuronGroup(OT, model=lif_eqs, threshold=-54*mV, reset=Vr)
NO2 = NeuronGroup(OT, model=lif_eqs, threshold=-54*mV, reset=Vr)

# NH = NeuronGroup(HT, model=eqs, threshold=threshold, reset=reset)
# NO1 = NeuronGroup(OT, model=eqs, threshold=threshold, reset=reset)
# NO2 = NeuronGroup(OT, model=eqs, threshold=threshold, reset=reset)

NI.Vm = Vr
NH.Vm = Vr
NO1.Vm = Vr
NO2.Vm = Vr

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

WHO1 = Connection(NH[0:HT/2], NO1, 'Vm', weight=rand(HT/2,len(NO1))*5*mV,structure='dense')
THO1 = Connection(NH[0:HT/2], NO1, 'K', weight= 0.0000001,structure='dense')
Pp_HO1 = NeuronGroup(HT/2, 'dPp/dt=-Pp/tauP:1')
Pm_HO1 = NeuronGroup(OT, 'dPm/dt=-Pm/tauP:1')

WHO2 = Connection(NH[HT/2:HT], NO2, 'Vm', weight=rand(HT/2,len(NO2))*5*mV,structure='dense')
THO2 = Connection(NH[HT/2:HT], NO2, 'K', weight= 0.0000001,structure='dense')
Pp_HO2 = NeuronGroup(HT/2, 'dPp/dt=-Pp/tauP:1')
Pm_HO2 = NeuronGroup(OT, 'dPm/dt=-Pm/tauP:1')

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

def pre_update_HO1(spikes):
    if len(spikes):
        Pp_HO1.Pp[spikes] += Ap
        for i in spikes:
            THO1.W[i, :] += Pm_HO1.Pm

def post_update_HO1(spikes):
    if len(spikes):
        Pm_HO1.Pm[spikes] += Am
        for i in spikes:
            THO1.W[:, i] += Pp_HO1.Pp

def pre_update_HO2(spikes):
    if len(spikes):
        Pp_HO2.Pp[spikes] += Ap
        for i in spikes:
            THO2.W[i, :] += Pm_HO2.Pm

def post_update_HO2(spikes):
    if len(spikes):
        Pm_HO2.Pm[spikes] += Am
        for i in spikes:
            THO2.W[:, i] += Pp_HO2.Pp

for i in range(inputs):
    SM_Pp_IH.append(SpikeMonitor(I[i], function=pre_update_IH_builder(i)))
    SM_Pm_IH.append(SpikeMonitor(NH, function=post_update_IH_builder(i)))
    SM_Pp_EH.append(SpikeMonitor(E[i], function=pre_update_EH_builder(i)))
    SM_Pm_EH.append(SpikeMonitor(NH, function=post_update_EH_builder(i)))

SM_Pp_HO1 = SpikeMonitor(NH[0:HT/2], function=pre_update_HO1)
SM_Pm_HO1 = SpikeMonitor(NO1, function=post_update_HO1)

SM_Pp_HO2 = SpikeMonitor(NH[HT/2:HT], function=pre_update_HO2)
SM_Pm_HO2 = SpikeMonitor(NO2, function=post_update_HO2)

@network_operation() 
def reduce_trace():
    for i in range(inputs):
        TIH[i].W *= beta
        TEH[i].W *= beta
    THO1.W *= beta
    THO2.W *= beta

    for i in range(inputs):
        clip(WIH[i].W + np.dot(TIH[i].W,np.diag(np.repeat([reward1[-1],reward2[-1]],HT/2)))*dt*gamma, -5.*mV, 0.*mV, WIH[i].W)
        clip(WEH[i].W + np.dot(TEH[i].W,np.diag(np.repeat([reward1[-1],reward2[-1]],HT/2)))*dt*gamma, 0.*mV, 5.*mV, WEH[i].W)
    clip(WHO2.W + THO2.W*reward2[-1]*dt*gamma, 0*mV, 5*mV, WHO2.W)
    clip(WHO1.W + THO1.W*reward1[-1]*dt*gamma, 0*mV, 5*mV, WHO1.W)

# ------------------------------------------------------------------------------------------------------------------------------

def policy1(spikes):
    if 0 in spikes:
        AB[0] += 1
        if case == 0:
            reward1.append(1)
        else:
            reward1.append(-1)
    else:
        reward1.append(0)

def policy2(spikes):
    if 0 in spikes:
        AB[1] += 1
        if case == 0:
            reward2.append(-1)
        else:
            reward2.append(1)
    else:
        reward2.append(0)

RM1 = SpikeMonitor(NO1,function=policy1)
RM2 = SpikeMonitor(NO2,function=policy2)

AB = [0,0]
reward1 = []
reward2 = []
rewards = []

for x in range(100):
    print 'E',x+1,'-',

    S[0].rate = 100*Hz
    S[1].rate = 0*Hz
    S[2].rate = 100*Hz
    S[3].rate = 0*Hz
    case = 0
    run(500*ms)
    print AB, AB.index(max(AB)),
    AB = [0,0]

    S[0].rate = 100*Hz
    S[1].rate = 0*Hz
    S[2].rate = 0*Hz
    S[3].rate = 100*Hz
    case = 1
    run(500*ms)
    print AB, AB.index(max(AB)),
    AB = [0,0]

    S[0].rate = 0*Hz
    S[1].rate = 100*Hz
    S[2].rate = 100*Hz
    S[3].rate = 0*Hz
    case = 1
    run(500*ms)
    print AB, AB.index(max(AB)),
    AB = [0,0]

    S[0].rate = 0*Hz
    S[1].rate = 100*Hz
    S[2].rate = 0*Hz
    S[3].rate = 100*Hz
    case = 0
    run(500*ms)
    print AB, AB.index(max(AB))
    AB = [0,0]

    rewards.append(sum(reward1[-Ns*4:])+sum(reward2[-Ns*4:]))

plot(rewards)
show()

gamma = 0

for x in range(10):
    print 'T',x+1,'-',

    S[0].rate = 100*Hz
    S[1].rate = 0*Hz
    S[2].rate = 100*Hz
    S[3].rate = 0*Hz
    case = 0
    run(500*ms)
    print AB, AB.index(max(AB)),
    AB = [0,0]

    S[0].rate = 100*Hz
    S[1].rate = 0*Hz
    S[2].rate = 0*Hz
    S[3].rate = 100*Hz
    case = 1
    run(500*ms)
    print AB, AB.index(max(AB)),
    AB = [0,0]

    S[0].rate = 0*Hz
    S[1].rate = 100*Hz
    S[2].rate = 100*Hz
    S[3].rate = 0*Hz
    case = 1
    run(500*ms)
    print AB, AB.index(max(AB)),
    AB = [0,0]

    S[0].rate = 0*Hz
    S[1].rate = 100*Hz
    S[2].rate = 0*Hz
    S[3].rate = 100*Hz
    case = 0
    run(500*ms)
    print AB, AB.index(max(AB))
    AB = [0,0]