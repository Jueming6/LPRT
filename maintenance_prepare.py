import numpy as np

class maintenance_pre():
    def __init__(self, n_condition, n_operation):
        self.n_condition = n_condition
        self.n_operation = n_operation

    ################################ maintenance performance ######################
    def repair_i(self, cc):
        repair_m = np.zeros([self.n_condition, self.n_condition])
        for i in range(self.n_condition):
            if i == 0:
                repair_m[i, 0] = 1
            else:
                repair_m[i, i] = (cc - 1) / cc
                for j in range(i):
                    repair_m[i, j] = (j + 1) / ((i + 1) * i / 2) * (1 - (cc - 1) / cc)  # sum along row = 1
        return repair_m

    def prep(self):
        natural_m = np.identity(self.n_condition)

        replace_m = np.zeros([self.n_condition, self.n_condition])
        replace_m[:, 0] = 0.99
        replace_m[:, 1] = 0.01

        rcc = np.random.randint(2, 12, self.n_operation - 2)  ##repair performance
        costcc = rcc * 10  ##larger rcc, better repair performance
        # print(rcc, costcc)

        repair_m = np.zeros([self.n_operation - 2, self.n_condition, self.n_condition])
        for i in range(self.n_operation - 2):
            repair_m[i, :, :] = self.repair_i(rcc[i])

        ############ ex. con2_m_performance = np.asarray([[0, 1, 0], [0.99, 0.01, 0]], [0.5, 0.5, 0]]) #######
        ########### natrural, replace, other operations ##############################
        con_m_performance = np.zeros([self.n_condition, self.n_operation, self.n_condition])  # sum along row = 1
        for i in range(self.n_condition):
            for j in range(self.n_operation):
                if j == 0:
                    con_m_performance[i, j, :] = natural_m[i, :]
                if j == 1:
                    con_m_performance[i, j, :] = replace_m[i, :]
                if j > 1:
                    con_m_performance[i, j, :] = repair_m[j - 2, i, :]

        ############ ex. np.asarray([[0.7, 0.3, 0], [0, 0.8, 0.2], [0, 0, 1]]) #######################
        degradeP = np.zeros([self.n_condition, self.n_condition])  # sum along row = 1
        for i in range(self.n_condition - 2):
            cc = np.random.randint(2, 12)
            for j in range(i, self.n_condition - 1):
                if j == i:
                    degradeP[i, j] = (cc - 1) / cc
                else:
                    degradeP[i, j] = (self.n_condition - j) / (
                            self.n_condition * (self.n_condition - 1 - i - 1) - (i + self.n_condition - 1) * (
                            self.n_condition - i - 2) / 2) * (1 - (cc - 1) / cc)
        degradeP[self.n_condition - 1, self.n_condition - 1] = 1  # worst condition degarde = 1
        degradeP[self.n_condition - 2, self.n_condition - 2] = 0.9
        degradeP[self.n_condition - 2, self.n_condition - 1] = 0.1

        ############# np.asarray([[0, 0, 0], [30, 50, 100], [500, 500, 1000]]) ########################
        maintenance_cost = np.zeros([self.n_operation, self.n_condition])
        # replace is the most expensive
        for i in range(1, self.n_operation):
            if i == 1:
                ccc = 150
            else:
                ccc = costcc[i - 2]

            for j in range(self.n_condition):
                maintenance_cost[i, j] = ccc * (j + 1) * 10

        ############ (maintenance transition & degradadtion trainsition) (combine & ravel) ########################
        prob = np.zeros([self.n_condition, self.n_operation, self.n_condition])
        prob_operation = np.zeros([self.n_condition, self.n_condition * self.n_operation])

        for i in range(self.n_condition):
            prob[i, :] = con_m_performance[i, :].dot(degradeP)

        for i in range(self.n_condition):
            for j in range(self.n_condition):
                prob_operation[i, j * self.n_operation: (j + 1) * self.n_operation] = prob[j, :, i]

        ############# ravel cost  ex. [0, 30, 500, 0, 50, 500, 0, 100, 1000] ###########################################

        mc = np.zeros(self.n_condition * self.n_operation)
        for i in range(self.n_condition):
            mc[i * self.n_operation: (i + 1) * self.n_operation] = maintenance_cost[:, i]
        
        return con_m_performance, prob, prob_operation, mc, degradeP