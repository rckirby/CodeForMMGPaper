# maps the generated data from heat equation into CSV files
# that are processed in the latex file to generate Figures 5.3, 5.4

import pickle

with open("heatgmres.dat", "rb") as f:
    results = pickle.load(f)

degs = (1, 2)
cfls = (1, 8)
levels = (1, 2, 3)
stages = (1, 2, 3, 4, 5)

for deg in degs:
    for cfl in cfls:
        with open(f"heat.deg{deg}.cfl{cfl}.csv", "w") as f:
            headers = "NS" + ",".join([f"Lev{l}it,Lev{l}time" for l in levels]) + "\n"
            f.write(headers)
            for s in stages:
                stuff = [results[(deg, s, l, cfl)] for l in levels]
                vals = ",".join([f"{x[1]},{x[0]}" for x in stuff]) 
                row = f"{s},{vals}\n"
                f.write(row)

