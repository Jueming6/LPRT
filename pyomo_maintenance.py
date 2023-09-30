from pyomo.environ import *
import numpy as np

model = ConcreteModel()
TT = 5#5
model.T = RangeSet(0, TT-1) # time periods

# i0 = 5.0 # initial inventory
# c = 4.6 # setup cost
# h_pos = 0.7 # inventory holding cost
# h_neg = 1.2 # shortage cost
# P = 5.0 # maximum production amount
# # demand during period t
# d = {1: 5.0, 2:7.0, 3:6.2, 4:3.1, 5:1.7}

n_condition = 3
n_operation = 3
model.n_condition = RangeSet(0, n_condition-1)
model.n_operation = RangeSet(0, n_operation-1)


N = 1000

s0 = [0.25, 0.7, 0.05]

con1_m_performance = np.asarray([[1, 0, 0], [1, 0, 0], [0.99, 0.01, 0]])
con2_m_performance = np.asarray([[0, 1, 0], [0.5, 0.5, 0], [0.99, 0.01, 0]])
con3_m_performance = np.asarray([[0, 0, 1], [0.1, 0.3, 0.6], [0.99, 0.01, 0]])
con_m_performance = np.zeros([n_condition, n_operation, n_condition])
con_m_performance[0, :] = con1_m_performance
con_m_performance[1, :] = con2_m_performance
con_m_performance[2, :] = con3_m_performance

degradeP = np.asarray([[0.7, 0.3, 0], [0, 0.8, 0.2], [0, 0, 1]])
maintenance_cost = np.asarray([[0, 0, 0], [30, 50, 100], [500, 500, 1000]])

# define the variables
#model.y = Var(model.T, domain=Binary)
#model.x = Var(model.T, domain=NonNegativeReals)
model.x = Var(RangeSet(TT), RangeSet(n_condition * (n_operation - 1)), initialize=0.6, bounds=(0, 1))    #action
#model.x = Set(initialize=1.01*np.zeros([model.T, (n_condition * (n_operation - 1))]), domain=NonNegativeReals, bounds = (0, 1), ordered = True)    #action
model.u = Var(RangeSet(TT),RangeSet(n_condition * n_operation), initialize = 0, within=NonNegativeReals)    #action
#model.s = Var(RangeSet(TT), RangeSet(n_condition), domain=NonNegativeReals)    #state
model.costi = Var(RangeSet(TT), initialize=0, domain=NonNegativeReals)  #cost
model.ii = Var(RangeSet(TT), RangeSet(n_condition), initialize = 0, domain=NonNegativeReals)
model.iii = Var(RangeSet(TT), initialize = 0, domain=NonNegativeReals)
#model.aa = Var(RangeSet(1), domain=NonNegativeReals)


uu = np.zeros(TT * n_operation * n_condition)
ss0 = np.zeros([TT + 1, n_condition])
ss0[0, :] = s0

#constraint
def time_action(m, t):
    #print(t)
    for tj in m.n_condition:
        #print(tj)
        aa = 0
        for tk in m.n_operation:
            #print(tk)
            if tk == 0:
                uu[np.ravel_multi_index([t, tj * n_operation + tk], [TT, n_operation * n_condition])] = \
                    value(m.x[t + 1, tj * (n_operation - 1) + tk + 1])
                m.u[t + 1, tj * n_operation + tk + 1] = \
                    value(m.x[t + 1, tj * (n_operation - 1) + tk + 1])
                aa += uu[np.ravel_multi_index([t, tj * n_operation + tk], [TT, n_operation * n_condition])]

            if tk < n_operation - 1 and tk > 0:
                uu[np.ravel_multi_index([t, tj * n_operation + tk], [TT, n_operation * n_condition])] = \
                    (m.x[t + 1, tj * (n_operation - 1) + tk + 1].value) * (1 - aa)
                m.u[t + 1, tj * n_operation + tk + 1] = \
                    (m.x[t + 1, tj * (n_operation - 1) + tk + 1].value) * (1 - aa)
                aa += uu[np.ravel_multi_index([t, tj * n_operation + tk], [TT, n_operation * n_condition])]

            if tk == n_operation - 1:
                uu[np.ravel_multi_index([t, tj * n_operation + tk], [TT, n_operation * n_condition])] = 1 - aa
                m.u[t + 1, tj * n_operation + tk + 1] = 1 - aa


    uu0 = uu[t * n_operation * n_condition : (t+1) * n_operation * n_condition]


    for ti in m.n_condition:
        ss0[t+1, :] += ((ss0[t, ti] * uu0[ti * n_operation : (ti + 1) * n_operation]).dot(con_m_performance[ti, :])).dot(degradeP)
        m.costi[t + 1] = value(m.costi[t + 1]) + N * uu0[ti * n_operation : (ti + 1) * n_operation].dot(maintenance_cost[:, ti])
    
    if ss0[t+1, n_condition-1] > 0.05:
        m.costi[t + 1] = value(m.costi[t + 1]) + 1e15
        
    for j in range(n_condition):
        m.ii[t+1, j+1] = ss0[t+1, j]

    return m.ii[t+1, n_condition] == ss0[t+1, n_condition-1]

model.action = Constraint(model.T, rule=time_action)


def last_condition(m, t):
    m.iii[t + 1] = m.ii[t+1, n_condition].value
    return m.iii[t + 1] -0.05 <= 0

model.last_con = Constraint(model.T, rule=last_condition)
#model.last_con_ct = Constraint(model.T, rule=last_condition_const)


# define the cost function
def obj_rule(m):
    return sum(m.costi[ti + 1] for ti in m.T)

model.obj = Objective(rule=obj_rule)

# solve the problem
import cplex
import sys
sys.path.append('/opt/ibm/ILOG/CPLEX_Studio_Community129/cplex/bin/x86-84_linux')
solver = SolverFactory('cplex', executable = "/opt/ibm/ILOG/CPLEX_Studio_Community129/cplex/bin/x86-64_linux/cplex")#('glpk')
solution = solver.solve(model) #, executable = "/opt/ibm/ILOG/CPLEX_Studio_Community129/cplex/bin/x86-64_linux/cplex")
model.action.pprint()
model.last_con.pprint()

from pyomo.opt import SolverStatus, TerminationCondition
if (solution.solver.status == SolverStatus.ok) and (solution.solver.termination_condition == TerminationCondition.optimal):
    print("Solution is feasible and optimal")
    print("Objective function value = ", model.obj())
elif solution.solver.termination_condition == TerminationCondition.infeasible:
    print ("Failed to find solution.")
else:
    # something else is wrong
    print(str(solution.solver))
# print the results
for t in model.T:
    #print(model.x[2, 3].value)
    print('Period: {0}, Prod. Amount: {1}'.format(t, uu[np.ravel_multi_index([t, 0], [TT, n_operation * n_condition]): (np.ravel_multi_index([t, n_operation * n_condition-1], [TT, n_operation * n_condition])+1)]))