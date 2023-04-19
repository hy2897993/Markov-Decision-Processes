#%%
import sys

sys.path.append('./burlap.jar')
import java
from collections import defaultdict
from time import clock
import csv
from collections import deque
from burlap.assignment4 import BasicGridWorld;
from burlap.behavior.singleagent.learning.tdmethods import QLearning;
from burlap.behavior.singleagent.planning.stochastic.policyiteration import PolicyIteration;
from burlap.behavior.singleagent.planning.stochastic.valueiteration import ValueIteration;
from burlap.oomdp.singleagent.environment import SimulatedEnvironment;
from burlap.oomdp.statehashing import SimpleHashableStateFactory;
from burlap.assignment4.util import MapPrinter;
from burlap.assignment4.util import BasicRewardFunction;
from burlap.assignment4.util import BasicTerminalFunction;
from burlap.assignment4.util import MapPrinter;
from burlap.oomdp.core import TerminalFunction;
from burlap.assignment4.util.AnalysisRunner import calcRewardInEpisode, simpleValueFunctionVis, getAllStates



def createCSVfile(n, convergence, rewards,  steps,  times,type, method):
    
    fname = 'files/small_MDP/{}_{}.csv'.format(type, method)

    with open(fname, 'w') as f:
        f.write('iter,time,reward,steps,convergence\n')
        writer = csv.writer(f, delimiter=',')
        writer.writerows(zip(range(1, n + 1), times, rewards, steps, convergence))


#%%
if __name__ == '__main__':

    d = 'small'
    discount = 0.90

    itera = 100;
    userMap = [[-1, -1, -1, -1, 100],
               [-1,  1,  1,  1,-100],
               [-1, -1,  1, -1, 100],
               [-1,  1,  1,  1, 100],
               [-1, -1, -1, -1, 100],]

    n = len(userMap)
    tmp = java.lang.reflect.Array.newInstance(java.lang.Integer.TYPE, [n, n])
    for i in range(n):
        for j in range(n):
            tmp[i][j] = userMap[i][j]
    userMap = MapPrinter().mapToMatrix(tmp)
    maxX = maxY = n - 1

    gen = BasicGridWorld(userMap, maxX, maxY)
    domain = gen.generateDomain()
    initialState = gen.getExampleState(domain);

    rf = BasicRewardFunction(maxX, maxY, userMap)
    tf = BasicTerminalFunction(maxX, maxY)
    env = SimulatedEnvironment(domain, rf, tf, initialState);


    hashingFactory = SimpleHashableStateFactory()
    time = defaultdict(list)
    rewards = defaultdict(list)
    steps = defaultdict(list)
    convergence = defaultdict(list)
    allStates = getAllStates(domain, rf, tf, initialState)
    
    
    
    
    #Value Iteration
    
    vi = ValueIteration(domain, rf, tf, discount, hashingFactory, -1, 1);
    vi.performReachabilityFrom(initialState)
    
    time['Value'].append(0)
    for n in range(1, 150):
        startTime = clock()
        vi.runVI()
        time['Value'].append(time['Value'][-1] + clock() - startTime)
        p = vi.planFromState(initialState);
        convergence['Value'].append(vi.latestDelta)


        r = []
        s = []
        for j in range(itera):
            ea = p.evaluateBehavior(initialState, rf, tf, 100);
            r.append(calcRewardInEpisode(ea))
            s.append(ea.numTimeSteps())
        rewards['Value'].append(sum(r) / float(len(r)))
        steps['Value'].append(sum(s) / float(len(s)))

        if vi.latestDelta < 1e-6:
            simpleValueFunctionVis(vi, p, initialState, domain, hashingFactory, "Value Iteration {}".format(n))
            break
    
    createCSVfile(n, convergence['Value'], rewards['Value'], steps['Value'], time['Value'][1:],  d, 'Value')




   #Value Iteration

    pi = PolicyIteration(domain, rf, tf, discount, hashingFactory, 1e-3, 10, 1)
    
    time['Policy'].append(0)
    for n in range(1, 150):
        startTime = clock()
        p = pi.planFromState(initialState);
        time['Policy'].append(time['Policy'][-1] + clock() - startTime)
        policy = pi.getComputedPolicy()
        current_policy = {state: policy.getAction(state).toString() for state in allStates}
        convergence['Policy2'].append(pi.lastPIDelta)
        if n == 1:
            convergence['Policy'].append(1000)
        else:
            changes = 0
            for k in last_policy.keys():
                if last_policy[k] != current_policy[k]:
                    changes += 1
            convergence['Policy'].append(changes)
        last_policy = current_policy

        r = []
        s = []
        for j in range(itera):
            ea = p.evaluateBehavior(initialState, rf, tf, 500)
            r.append(calcRewardInEpisode(ea))
            s.append(ea.numTimeSteps())
        rewards['Policy'].append(sum(r) / float(len(r)))
        steps['Policy'].append(sum(s) / float(len(s)))

        if convergence['Policy2'][-1] < 1e-6:
            simpleValueFunctionVis(pi, p, initialState, domain, hashingFactory, "Policy Iteration {}".format(n))
            break

    createCSVfile(n, convergence['Policy2'],rewards['Policy'], steps['Policy'],  time['Policy'][1:], d, 'Policy')

    #Q-learning
    iterations = range(1, 200)
    for lr in [0.01, 0.1, 0.3, 0.5, 0.9, 0.99]:
        for q in [-100, 0, 100]:
            for e in [0.1, 0.3, 0.5, 0.7, 0.9]:
                last10Chg = deque([99] * 10, maxlen=10)
                Q_file = 'Q-Learning L{:0.2f} q{:0.1f} E{:0.1f}'.format(lr, q, e)
                agent = QLearning(domain, discount, hashingFactory, q, lr, e, 300)

                for n in iterations:
                
                    startTime = clock()
                    ea = agent.runLearningEpisode(env, 300)
                    if len(time[Q_file]) > 0:
                        time[Q_file].append(time[Q_file][-1] + clock() - startTime)
                    else:
                        time[Q_file].append(clock() - startTime)
                    env.resetEnvironment()
                    agent.initializeForPlanning(rf, tf, 1)
                    p = agent.planFromState(initialState)  
                    last10Chg.append(agent.maxQChangeInLastEpisode)
                    convergence[Q_file].append(sum(last10Chg) / 10.)


                    r = []
                    s = []
                    for j in range(itera):
                        ea = p.evaluateBehavior(initialState, rf, tf, 500)
                        r.append(calcRewardInEpisode(ea))
                        s.append(ea.numTimeSteps())
                    rewards[Q_file].append(sum(r) / float(len(r)))
                    steps[Q_file].append(sum(s) / float(len(s)))



                    if convergence[Q_file][-1] < 0.25:
                        simpleValueFunctionVis(agent, p, initialState, domain, hashingFactory,"Q-Learning Iteration {}".format(n))
                        break
                print("\n end")
                
                createCSVfile(n, convergence[Q_file], rewards[Q_file], steps[Q_file], time[Q_file], d, Q_file)
