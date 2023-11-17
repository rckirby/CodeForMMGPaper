# Computes time and iteration counts for solving eddy current problem with GMRES
# and dumps the results to a file.  
import pickle
from copy import deepcopy

import numpy
from firedrake import (PCG64, Constant, DistributedMeshOverlapType, Function,
                       FunctionSpace, MeshHierarchy, RandomGenerator,
                       TestFunction, UnitCubeMesh, dx, curl, inner, prolong)
from firedrake.petsc import PETSc
from irksome import Dt, RadauIIA, TimeStepper
from irksome.tools import IA
from mpi4py import MPI

PETSc.Log().begin()
def get_time(event, comm=MPI.COMM_WORLD):
    return comm.allreduce(PETSc.Log.Event(event).getPerfInfo()["time"],
                          op=MPI.SUM) / comm.size


dist_params = {"partition": True,
               "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}

rg = RandomGenerator(PCG64(seed=123456789))


params = {
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "ksp_gmres_restart": 20,
    "ksp_monitor": None,
    "ksp_rtol": 1.e-8,
    "pc_type": "mg",
    "mg_levels": {
        "ksp_type": "chebyshev",
        "ksp_convergence_test": "skip",
        "ksp_max_it": 2,
        "pc_type": "python",
        "pc_python_type": "firedrake.ASMStarPC",
        "pc_star_construct_dim": 0,
        "pc_star_backend_type": "tinyasm"
        },
    "mg_coarse": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }
}


def get_random(mh, deg):
    return rg.uniform(FunctionSpace(mh[-1], "N1curl", deg), -1, 1)


def time_its(N_base, levels, cfl, deg, bt):
    PETSc.Sys.Print(N_base, levels, cfl, deg, bt)
    msh_base = UnitCubeMesh(N_base, N_base, N_base,
                            distribution_parameters=dist_params)
    mh = MeshHierarchy(msh_base, levels)
    msh = mh[-1]

    V = FunctionSpace(msh, "N1curl", deg)

    AA = get_random(mh, deg)
    v = TestFunction(V)

    t = Constant(0)
    dt = Constant(cfl / N_base / 2**levels)

    F = inner(Dt(AA), v) * dx + inner(curl(AA), curl(v)) * dx
    
    if deg == 1:
        params["mg_levels"]["ksp_chebyshev_esteig"] = "0.0,0.1,0,1.05"
    elif deg == 2:
        params["mg_levels"]["ksp_chebyshev_esteig"] = "0.0,0.25,0,1.2"

    stepper = TimeStepper(F, bt, t, dt, AA,
                          splitting=IA,
                          solver_parameters=params)

    sn = f"{N_base}{levels}{cfl}{deg}{str(bt)}{cfl}"
    PETSc.Sys.Print(sn)
    with PETSc.Log.Stage(sn):
        stepper.advance()
        snes = get_time("KSPSolve")

    return stepper.solver.snes.getKSP().getIterationNumber(), snes


N_base = 4
levels = 3

stages = (1, 2, 3, 4, 5)
degs = (1, 2)
cfls = (1, 8)

results = {}

for deg in degs:
    for ns in stages:
        bt = RadauIIA(ns)
        for lev in range(1, levels+1):
            for cfl in cfls:
                its, tm = time_its(N_base, levels, cfl, deg, bt)
                x = f"k {deg}, RIIA({ns}), cfl {cfl}: T {tm}, Its {its}"
                PETSc.Sys.Print(x)
                results[(deg, ns, lev, cfl)] = tm, its

if MPI.COMM_WORLD.rank == 0:
    print(results)
    with open("eddygmres.dat", "wb") as f:
        pickle.dump(results, f)
