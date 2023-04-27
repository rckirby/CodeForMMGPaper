# Generates data for eigenvalue plots used in Figure 3.1
import numpy as np
from irksome import GaussLegendre, RadauIIA


def nm(x):
    if isinstance(x, GaussLegendre):
        return f"GL{x.num_stages}"
    elif isinstance(x, RadauIIA):
        return f"RIIA{x.num_stages}"


for btcons in [RadauIIA, GaussLegendre]:
    for k in range(1, 11):
        bt = btcons(k)
        evs = np.linalg.eigvals(bt.A)
        with open(f"{nm(bt)}eigs.csv", "w") as f:
            f.write("Re,Im\n")
            for x in evs:
                f.write(f"{x.real},{x.imag}\n")
