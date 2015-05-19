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
gamma = 0.01
Ap = 1
Am = -1
lif_eqs = Equations('''
dVm/dt=(Vl-Vm)/tau : volt
K : 1
''')

# ------------------------------------------------------------------------------------------------------------------------------

inputs = 4
outputs = 2
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
WHO2 = []
THO2 = []
Pp_HO2 = []
Pm_HO2 = []

for i in range(outputs):
    WHO.append(Connection(NH[(HT/outputs)*i:(HT/outputs)*(i+1)], NO[i], 'Vm', weight=rand(HT/outputs,len(NO[i]))*5*mV,structure='dense'))
    THO.append(Connection(NH[(HT/outputs)*i:(HT/outputs)*(i+1)], NO[i], 'K', weight= 0.0000001,structure='dense'))
    Pp_HO.append(NeuronGroup(HT/outputs, 'dPp/dt=-Pp/tauP:1'))
    Pm_HO.append(NeuronGroup(OT, 'dPm/dt=-Pm/tauP:1'))
    WHO2.append(Connection(NH[(HT/outputs)*i:(HT/outputs)*(i+1)], NO[1-i], 'Vm', weight=rand(HT/outputs,len(NO[1-i]))*5*mV,structure='dense'))
    THO2.append(Connection(NH[(HT/outputs)*i:(HT/outputs)*(i+1)], NO[1-i], 'K', weight= 0.0000001,structure='dense'))
    Pp_HO2.append(NeuronGroup(HT/outputs, 'dPp/dt=-Pp/tauP:1'))
    Pm_HO2.append(NeuronGroup(OT, 'dPm/dt=-Pm/tauP:1'))

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

def pre_update_HO2_builder(n):
    def pre_update_HO2(spikes):
        if len(spikes):
            Pp_HO2[n].Pp[spikes] += Ap
            for i in spikes:
                THO2[n].W[i, :] += Pm_HO2[n].Pm
    pre_update_HO2.__name__ = 'pre_update_HO2'+str(n)
    return pre_update_HO2

def post_update_HO2_builder(n):
    def post_update_HO2(spikes):
        if len(spikes):
            Pm_HO2[n].Pm[spikes] += Am
            for i in spikes:
                THO2[n].W[:, i] += Pp_HO2[n].Pp
    post_update_HO2.__name__ = 'post_update_HO2'+str(n)
    return post_update_HO2

for i in range(inputs):
    SM_Pp_IH.append(SpikeMonitor(I[i], function=pre_update_IH_builder(i)))
    SM_Pm_IH.append(SpikeMonitor(NH, function=post_update_IH_builder(i)))
    SM_Pp_EH.append(SpikeMonitor(E[i], function=pre_update_EH_builder(i)))
    SM_Pm_EH.append(SpikeMonitor(NH, function=post_update_EH_builder(i)))

SM_Pp_HO = []
SM_Pm_HO = []
SM_Pp_HO2 = []
SM_Pm_HO2 = []

for i in range(outputs):
    SM_Pp_HO.append(SpikeMonitor(NH[(HT/outputs)*i:(HT/outputs)*(i+1)], function=pre_update_HO_builder(i)))
    SM_Pm_HO.append(SpikeMonitor(NO[i], function=post_update_HO_builder(i)))
    SM_Pp_HO2.append(SpikeMonitor(NH[(HT/outputs)*i:(HT/outputs)*(i+1)], function=pre_update_HO2_builder(i)))
    SM_Pm_HO2.append(SpikeMonitor(NO[1-i], function=post_update_HO2_builder(i)))

@network_operation() 
def reduce_trace():
    for i in range(inputs):
        TIH[i].W *= beta
        TEH[i].W *= beta
    for i in range(outputs):
        THO[i].W *= beta
        THO2[i].W *= beta

    rews = []
    rsum[0] += sum(reward)
    for i in range(outputs):
        rews.append(reward[i])

    for i in range(inputs):
        clip(WIH[i].W + np.dot(TIH[i].W,np.diag(np.repeat(rews,HT/outputs)))*dt*gamma, -5*mV, 0*mV, WIH[i].W)
        clip(WEH[i].W + np.dot(TEH[i].W,np.diag(np.repeat(rews,HT/outputs)))*dt*gamma, 0*mV, 5*mV, WEH[i].W)

    for i in range(outputs):
        clip(WHO[i].W + THO[i].W*reward[i]*dt*gamma, 0*mV, 5*mV, WHO[i].W)
        clip(WHO2[i].W + THO2[i].W*reward[1-i]*dt*gamma, 0*mV, 5*mV, WHO2[i].W)

# ------------------------------------------------------------------------------------------------------------------------------
def policy_builder(n):
    def policy(spikes):
        if len(spikes):
            AB[n] += len(spikes)
            if ctype == n:
                reward[n] = myrew
            else:
                reward[n] = -myrew
        else:
            reward[n] = 0
    policy.__name__ = 'policy'+str(n)
    return policy

RM = []

for i in range(outputs):
    RM.append(SpikeMonitor(NO[i],function=policy_builder(i)))

cases = 4
AB = [0]*outputs
reward = [0]*outputs
rewards = []
rsum = [0]
myrew = 1
ract = [0]*cases
Rk = [0]*cases

for x in range(100):
    print 'E',x+1,'-',
    rcsum = 0

    S[0].rate = 100*Hz
    S[1].rate = 0*Hz
    S[2].rate = 100*Hz
    S[3].rate = 0*Hz
    ctype = 0
    case = 0
    run(500*ms)
    print AB, AB.index(max(AB)),
    AB = [0,0]
    rcsum += rsum[0]
    ract[case] = rsum[0]
    Rk[case] += (rsum[0] - Rk[case])/100.
    myrew = np.clip((ract[case] - Rk[case])/ract[case], 0, 1)
    rsum = [0]

    S[0].rate = 100*Hz
    S[1].rate = 0*Hz
    S[2].rate = 0*Hz
    S[3].rate = 100*Hz
    ctype = 1
    case = 1
    run(500*ms)
    print AB, AB.index(max(AB)),
    AB = [0,0]
    rcsum += rsum[0]
    ract[case] = rsum[0]
    Rk[case] += (rsum[0] - Rk[case])/100.
    myrew = np.clip((ract[case] - Rk[case])/ract[case], 0, 1)
    rsum = [0]

    S[0].rate = 0*Hz
    S[1].rate = 100*Hz
    S[2].rate = 100*Hz
    S[3].rate = 0*Hz
    ctype = 1
    case = 2
    run(500*ms)
    print AB, AB.index(max(AB)),
    AB = [0,0]
    rcsum += rsum[0]
    ract[case] = rsum[0]
    Rk[case] += (rsum[0] - Rk[case])/100.
    myrew = np.clip((ract[case] - Rk[case])/ract[case], 0, 1)
    rsum = [0]

    S[0].rate = 0*Hz
    S[1].rate = 100*Hz
    S[2].rate = 0*Hz
    S[3].rate = 100*Hz
    ctype = 0
    case = 3
    run(500*ms)
    print AB, AB.index(max(AB))
    AB = [0,0]
    rcsum += rsum[0]
    ract[case] = rsum[0]
    Rk[case] += (rsum[0] - Rk[case])/100.
    myrew = np.clip((ract[case] - Rk[case])/ract[case], 0, 1)
    rsum = [0]

    rewards.append(rcsum)

#print rewards
plot(rewards)
show()