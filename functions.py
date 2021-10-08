from pulp import LpVariable, LpInteger, value
from pulp import LpProblem, LpMinimize, LpStatus
from pulp import PULP_CBC_CMD
import numpy as np
import pandas as pd

def optimize(demand_data):
    all_bundle_variables = []
    all_missing_variables = []
    types = ["b1", "b2", "b3", "b4", "b5"]
    missing_weights = np.array([4] * 5)
    for product, df in demand_data.groupby("product"):
        problem = LpProblem("allocation", LpMinimize)
        total_demand = df["demand"].sum()
        bundle_variables = {}
        missing_variables = {}
        for type_ in types:
            bundle_variables[type_] = LpVariable(name=type_, lowBound=0, cat=LpInteger)
            missing_variables[type_] = LpVariable(
                name=f"missing_{type_}", lowBound=0, cat=LpInteger
            )
        bundle_weight_matrix = df[types].fillna(0).to_numpy()
        bundle_weights = bundle_weight_matrix.sum(axis=0)
        bundle_vector = np.array(list(bundle_variables.values()))
        n_types = df.shape[0]
        missing_weight_matrix = np.tile(missing_weights, (n_types, 1))
        missing_vector = np.array(list(missing_variables.values()))

        bundle_exprs = bundle_vector * bundle_weight_matrix

        for qty, bundle in zip(df["demand"], bundle_exprs):
            problem += sum(bundle) + sum(missing_vector) >= qty
        total_bundles_expr = bundle_weights * bundle_vector
        total_missing_expr = missing_weights * missing_vector
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

    allocation[types] = allocation[types].applymap(value)
    allocation["total"] = allocation[types].sum(axis=1)

    missing[types] = missing[types].applymap(value)
    missing["total"] = missing[types].sum(axis=1)

    return allocation, missing, problem