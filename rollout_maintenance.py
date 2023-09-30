import numpy as np
import time
import matplotlib.pyplot as plt
import random

#maintenance_cost = np.asarray([[0, 0, 0], [30, 50, 100], [500, 500, 1000]])


#p = np.asarray([[0.7, 0.3, 0], [0, 0.8, 0.2], [0, 0, 1]])


#natural_m = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#repair_m = np.asarray([[1, 0, 0], [0.5, 0.5, 0], [0.1, 0.3, 0.6]])
#replace_m = np.asarray([[0.99, 0.01, 0], [0.99, 0.01, 0], [0.99, 0.01, 0]])

#con1_m_performance = np.asarray([[1, 0, 0], [1, 0, 0], [0.99, 0.01, 0]])
#con2_m_performance = np.asarray([[0, 1, 0], [0.5, 0.5, 0], [0.99, 0.01, 0]])
#con3_m_performance = np.asarray([[0, 0, 1], [0.1, 0.3, 0.6], [0.99, 0.01, 0]])



###############  Mdeision = [natural, repair, replace] % #############################
def condition1_m(x, Mdecision1, con1_m_performance, p):
    afterm = (x[0] * Mdecision1).dot(con1_m_performance)
    afterp = afterm.dot(p)
    mcost = (N * x[0] * Mdecision1).dot (maintenance_cost[:, 0])
    return afterp, mcost


def condition2_m(x, Mdecision2, con2_m_performance, p):
    afterm = (x[1] * Mdecision2).dot(con2_m_performance)
    afterp = afterm.dot(p)
    mcost = (N * x[1] * Mdecision2) .dot (maintenance_cost[:, 1])
    return afterp, mcost


def condition3_m(x, Mdecision3, con3_m_performance, p):
    afterm = (x[2] * Mdecision3).dot(con3_m_performance)
    afterp = afterm.dot(p)
    mcost = (N * x[2] * Mdecision3) .dot (maintenance_cost[:, 2])
    return afterp, mcost


# x = np.asarray([0.2, 0.7, 0.1])
# Mdecision3 = np.asarray([0.5, 0.2, 0.3])
# bb,c = condition3_m(x, Mdecision3, con3_m_performance, p)
# print(bb,c)


def single_decision(N):
    possible_percentage = np.zeros(N) #N including 0, 1
    for i in range(N):
        possible_percentage[i] = 0 + 1/(N-1) * i

    NN = int(N * (N + 1) / 2)
    possible_action = np.zeros([NN, 3])
    for i in range(N):
        for j in range(N-i):
                possible_action[int(N * i - (i - 1) * i / 2 + j)] = [possible_percentage[i], possible_percentage[j], 1 - possible_percentage[i] - possible_percentage[j]]

    return possible_action


def uk_step(uk, xk, con1_m_performance, con2_m_performance, con3_m_performance, p):
    a1, b1 = condition1_m(xk, uk[0:3], con1_m_performance, p)
    a2, b2 = condition2_m(xk, uk[3:6], con2_m_performance, p)
    a3, b3 = condition3_m(xk, uk[6:9], con3_m_performance, p)
    reward = -(b1 + b2 + b3)
    if a1[2] + a2[2] + a3[2] > 0.05:
        reward = reward - 10e20
    #else:
        #reward += 10e7
    newxk = a1 + a2 + a3
    return newxk, reward

# uk = np.asarray([0, 1, 0, 0.2, 0.3, 0.5, 0.5, 0.5, 0])
# xk = np.asarray([0.2, 0.7, 0.1])
# bb, cc = uk_step(uk, xk, con1_m_performance, con2_m_performance, con3_m_performance, p)
# print(bb, cc)

def base_policy_Q(n_condition, n_operatiion, N, T, xk, uk, tk, agenti, possible_action, con1_m_performance, con2_m_performance, con3_m_performance, p, base_decision):
    Q0 = -1e100
    uk00 = np.zeros(n_condition * n_operatiion)
    uk00[:] = uk[:]
    xk0 = np.zeros(n_condition)
    xk0[:] = xk[:]
    uu = np.zeros(n_condition * n_operatiion)
    uu0 = np.zeros(n_condition * n_operatiion)

    for j in range(len(possible_action)):
        Q = 0
        uk[:] = uk00[:]
        uk[n_operatiion * agenti : n_operatiion * (agenti+1)] = possible_action[j]
        uu[:] = uk[:]
        xk[:] = xk0[:]

        for i in range(tk, T):
            xk, rk = uk_step(uk, xk, con1_m_performance, con2_m_performance, con3_m_performance, p)
            #xk = newxk
            Q += rk
            uk = base_decision
        #print(newxk)
        
        if Q > Q0:
            Q0 = Q
            uu0[:] = uu[:]
        
        
    return Q, uu0



# q, uu0, newxk0 = base_policy_Q(n_condition, N, T, xk, uk, tk, agenti, possible_action, con1_m_performance, con2_m_performance, con3_m_performance, p, base_decision)
# print(q, uu0, newxk0)


def control_u(n_condition, n_operatiion, N, T, x0, possible_action, con1_m_performance, con2_m_performance, con3_m_performance, p, base_decision):

    ###### control sequence
    u = np.zeros([T, n_condition * n_operatiion])
    ###### condition
    pp = np.zeros([T+1, n_condition])
    xk = x0
    pp[0] = x0
    rr = np.zeros(T)

    for ti in range(T):
        uk = base_decision
        for agentj in range(n_condition):
            Q, uu0 = base_policy_Q(n_condition, n_operatiion, N, T, pp[ti], uk, ti, agentj, possible_action, con1_m_performance, con2_m_performance, con3_m_performance, p, base_decision)
            #u[ti, n_operatiion * agentj : n_operatiion * (agentj+1)] = uu0[n_operatiion * agentj : n_operatiion * (agentj+1)]
            uk[n_operatiion * agentj : n_operatiion * (agentj+1)] = uu0[n_operatiion * agentj : n_operatiion * (agentj+1)]
        pp[ti+1], rr[ti] = uk_step(uk, pp[ti], con1_m_performance, con2_m_performance, con3_m_performance, p)
        u[ti] = uk
    return u, pp, rr



base_decision = np.asarray([0, 1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 1.0, 0]) #np.asarray([0, 1.0, 0, 0, 1.0, 0, 0, 1.0, 0])
T = 10  #8: t0~t8, 0~T
x0 = np.asarray([0.20, 0.70, 0.1]) #np.asarray([0.20, 0.70, 0.1])
p=np.asarray([[0.5, 0.5, 0], [0, 0.9, 0.1], [0, 0, 1]])
con1_m_performance = np.asarray([[1, 0, 0], [0.99, 0.01, 0], [1, 0, 0]])
con2_m_performance = np.asarray([[0, 1, 0], [0.99, 0.01, 0], [0.5, 0.5, 0]])
con3_m_performance = np.asarray([[0, 0, 1], [0.99, 0.01, 0], [0.1, 0.3, 0.6]])
maintenance_cost = np.asarray([[0, 0, 0], [1500, 3000, 4500], [300, 500, 900]])
N = 1000

possible_action = single_decision(11)
n_condition = 5 #3
n_operatiion = 3
u, pp, rr = control_u(n_condition, n_operatiion, N, T, x0, possible_action, con1_m_performance, con2_m_performance,
                  con3_m_performance, p, base_decision)

#np.savetxt('u', u, delimiter=', ', fmt='%1.3f')
#np.savetxt('pp', pp, delimiter=', ', fmt='%1.3f')
#np.savetxt('rr', rr, delimiter=', ', fmt='%1.3f')
print(u, np.round(pp, 3), rr, sum(rr))
for i in range(len(rr)):
    print(-int(rr[i]))
print(-sum(rr))
#np.savetxt('cost_rollout', rr)
#np.savetxt('3c200', cost,  delimiter=', ', fmt='%1.3f')
#np.savetxt('3jjj200', jjj,  delimiter=', ', fmt='%1.3f')
#print(cost, obs, pp, u, nn, jcost, prob, jjj)