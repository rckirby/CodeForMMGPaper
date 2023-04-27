# This writes out a simple CSV file used to generate Figure 3.2

from irksome import GaussLegendre, RadauIIA
from numpy.linalg import cond, eig

bts = [RadauIIA, GaussLegendre]

for nm, bt in zip(["RadauIIA", "GaussLegendre"], bts):
    with open(f"{nm}cond.csv", "w") as f:
        f.write("k,kappa\n")
        for k in range(1, 11):
            A = bt(k).A
            f.write(f"{k}, {cond(eig(A)[1])}\n")
