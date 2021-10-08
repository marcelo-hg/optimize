from pulp import LpVariable, LpInteger, value
from pulp import LpProblem, LpMinimize, LpStatus
from pulp import PULP_CBC_CMD
import numpy as np
import pandas as pd

class Optimizer:

    def __init__(self, data, missing_weghts, types):
        self.data = data
        self.missing_weights = missing_weghts
        self.types = types

    def run(self):
        all_bundle_variables = []
        all_missing_variables = []
        for product, df in self.data.groupby("product"):
            problem = LpProblem("allocation", LpMinimize)
            total_demand = df["demand"].sum()
            bundle_variables = {}
            missing_variables = {}
            for type_ in self.types:
                bundle_variables[type_] = LpVariable(name=type_, lowBound=0, cat=LpInteger)
                missing_variables[type_] = LpVariable(
                    name=f"missing_{type_}", lowBound=0, cat=LpInteger
                )
            bundle_weight_matrix = df[self.types].fillna(0).to_numpy()
            bundle_weights = bundle_weight_matrix.sum(axis=0)
            bundle_vector = np.array(list(bundle_variables.values()))
            n_types = df.shape[0]
            missing_weight_matrix = np.tile(self.missing_weights, (n_types, 1))
            missing_vector = np.array(list(missing_variables.values()))

            bundle_exprs = bundle_vector * bundle_weight_matrix

            for qty, bundle in zip(df["demand"], bundle_exprs):
                problem += sum(bundle) + sum(missing_vector) >= qty
            total_bundles_expr = bundle_weights * bundle_vector
            total_missing_expr = self.missing_weights * missing_vector
            problem += sum(total_bundles_expr) <= 2 * total_demand

            OBJECTIVE = sum(total_bundles_expr) + sum(total_missing_expr)
            problem += OBJECTIVE
            status = problem.solve(PULP_CBC_CMD(msg=False))
            bundle_variables["product"] = product
            missing_variables["product"] = product
            all_bundle_variables.append(bundle_variables)
            all_missing_variables.append(missing_variables)
        print(LpStatus[problem.status])
        allocation = pd.DataFrame(all_bundle_variables)
        missing = pd.DataFrame(all_missing_variables)

        allocation[self.types] = allocation[self.types].applymap(value)
        allocation["total"] = allocation[self.types].sum(axis=1)

        missing[self.types] = missing[self.types].applymap(value)
        missing["total"] = missing[self.types].sum(axis=1)

        return allocation, missing, problem