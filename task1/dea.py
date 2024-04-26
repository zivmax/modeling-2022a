import numpy as np
from scipy.optimize import fmin_slsqp


class DEA(object):

    def __init__(self, inputs, outputs):
        """
        Initialize the DEA object with input data
        n = number of entities (observations)
        m = number of inputs (variables, features)
        r = number of outputs
        :param inputs: inputs, n x m numpy array
        :param outputs: outputs, n x r numpy array
        :return: self
        """

        # supplied data
        self.inputs = inputs
        self.outputs = outputs

        # parameters
        self.j = inputs.shape[0]
        self.n = inputs.shape[1]
        self.m = outputs.shape[1]

        # result arrays
        self.efficiency = np.zeros(self.j)  # thetas

        # names
        self.names = []

    def __efficiency(self, x, unit):
        """
        Theta target function for one unit
        :param x: combined weights
        :param unit: which production unit to compute
        :return: theta
        """
        in_w, out_w = x[:self.n], x[self.n:(self.n+self.m)]  # unroll the weights
        denominator = np.dot(self.inputs[unit], in_w)
        numerator = np.dot(self.outputs[unit], out_w)

        return 1 - numerator/denominator

    def __constraints(self, x, unit):
        """
        Constraints for optimization for one unit
        :param x: combined weights
        :param unit: which production unit to compute
        :return: array of constraints
        """

        lambdas = x[(self.n+self.m):]
        constr = []  # init the constraint array

        # for each input, lambdas with inputs
        for input in range(self.n):
            t = self.__efficiency(x, unit)
            lhs = np.dot(self.inputs[:, input], lambdas)
            cons = t * self.inputs[unit, input] - lhs
            constr.append(cons)

        # for each output, lambdas with outputs
        for output in range(self.m):
            lhs = np.dot(self.outputs[:, output], lambdas)
            cons = lhs - self.outputs[unit, output]
            constr.append(cons)

        # for each unit
        for u in range(self.j):
            constr.append(lambdas[u])

        return np.array(constr)

    def __optimize(self):
        """
        Optimization of the DEA model
        Use: http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.linprog.html
        A = coefficients in the constraints
        b = rhs of constraints
        c = coefficients of the target function
        :return:
        """
        d0 = self.n + self.m + self.j
        # iterate over units
        for unit in range(self.j):
            # weights
            x0 = np.random.rand(d0) - 0.5
            x0 = fmin_slsqp(self.__efficiency, x0, f_ieqcons=self.__constraints, args=(unit,))
            # unroll weights
            self.efficiency[unit] = self.__efficiency(x0, unit)

    def name_units(self, names):
        """
        Provide names for units for presentation purposes
        :param names: a list of names, equal in length to the number of units
        :return: nothing
        """

        assert (self.j == len(names))

        self.names = names

    def fit(self):
        """
        Optimize the dataset, generate basic table
        :return: table
        """

        self.__optimize()  # optimize

        print("Final thetas for each unit:\n")
        print("---------------------------\n")
        for n, eff in enumerate(self.efficiency):
            if len(self.names) > 0:
                name = "Unit %s" % self.names[n]
            else:
                name = "Unit %d" % (n+1)
            print("%s theta: %.4f" % (name, eff))
            print("\n")
        print("---------------------------\n")