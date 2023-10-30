import pickle

with open("stokesgmres.dat", "rb") as f:
    results = pickle.load(f)

cfls = (1, 8)
levels = (1, 2, 3)
stages = (1, 2, 3, 4, 5)

for cfl in cfls:
    with open(f"stokes.cfl{cfl}.csv", "w") as f:
        headers = "NS" + ",".join([f"Lev{l}it,Lev{l}time" for l in levels]) + "\n"
        f.write(headers)
        for s in stages:
            stuff = [results[(s, l, cfl)] for l in levels]
            vals = ",".join([f"{x[0]},{x[1]}" for x in stuff]) 
            row = f"{s},{vals}\n"
            f.write(row)
