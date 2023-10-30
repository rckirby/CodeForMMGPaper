# Code used to estimate contraction constant for MG iteration
# for multi-stage heat equation
import gc
from copy import deepcopy

import numpy
from firedrake import (PCG64, Constant, DirichletBC,
                       DistributedMeshOverlapType, Function,
                       FunctionSpace, Mesh, MeshHierarchy, RandomGenerator, SpatialCoordinate,
                       TestFunctions, VectorFunctionSpace,
                       as_vector, div, grad, dx, inner, prolong, split)
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


def get_random(mh):
    V0 = VectorFunctionSpace(mh[-1], "Q", 2)
    V1 = FunctionSpace(mh[-1], "DPC", 1)
    return rg.random(V0*V1)


def run(levels, cfl, bt):
    gc.collect()
    PETSc.Sys.Print(levels, cfl, bt)
    msh_base = Mesh("obst.quad.msh")
    mh = MeshHierarchy(msh_base, levels)
    msh = mh[-1]

    V0 = VectorFunctionSpace(mh[-1], "Q", 2)
    V1 = FunctionSpace(mh[-1], "DPC", 1)
    Z = V0 * V1

    AA = get_random(mh)

    v, w = TestFunctions(Z)
    x, y = SpatialCoordinate(msh)

    up = Function(Z)
    u, p = split(up)
    up.subfunctions[0].assign(AA.subfunctions[0])
    
    t = Constant(0)
    dt = Constant(cfl * 0.1 / 2**levels)

    F = (inner(Dt(u), v) * dx
         + inner(grad(u), grad(v)) * dx
         - inner(p, div(v)) * dx
         + inner(div(u), w) * dx)

    x, y = SpatialCoordinate(msh)

    Umax = Constant(10.0, domain=msh)
    H = Constant(0.41)

    inlet_expr = 4 * y * (H - y) * Umax
    # boundary conditions are specified for each subspace
    bcs = [DirichletBC(Z.sub(0),
                       as_vector([inlet_expr, 0]), (9,)),
           DirichletBC(Z.sub(0), Constant((0, 0)), (10, 12))]
    exclusions = ",".join([str(2*i+1) for i in range(bt.num_stages)])
    it_params = {"snes_type": "ksponly",
                 "ksp_type": "richardson",
                 "ksp_richardson_scale": 1.0,
                 "ksp_monitor": None,
                 "ksp_converged_reason": None,
                 "ksp_gmres_restart": 50,
                 "ksp_rtol": 1.e-8,
                 "pc_type": "mg",
                 "mg_levels": {
                     "ksp_type": "chebyshev",
                     "ksp_max_it": 2,
                     "ksp_convergence_test": "skip",
                     "pc_type": "python",
                     "pc_python_type": "firedrake.ASMVankaPC",
                     "pc_vanka_construct_codim": 0,
                     "pc_vanka_exclude_subspaces": exclusions,
                     "pc_vanka_backend_type": "tinyasm"},
                 "mg_coarse": {
                     "ksp_type": "preonly",
                     "pc_type": "lu",
                     "pc_factor_mat_solver_type": "mumps",
                     "mat_mumps_icntl_14": 200}
                 }

    its = {}

    stepper = TimeStepper(F, bt, t, dt, up,
                          splitting=IA,
                          solver_parameters=it_params)
    
    sn = f"{levels}{cfl}{str(bt)}{cfl}"
    PETSc.Sys.Print(sn)
    with PETSc.Log.Stage(sn):
        stepper.advance()
        snes = get_time("KSPSolve")

    return stepper.solver.snes.getKSP().getIterationNumber(), snes


levels = (1, 2, 3, 4)
stages = (1, 2, 3, 4, 5)
cfls = (1, 8)

results = {}

for ns in stages:
    bt = RadauIIA(ns)
    for lev in levels:
        for cfl in cfls:
            its, tm = run(lev, cfl, bt)
            x = f"RadauIIA({ns}), cfl {cfl}: Its {its}, tm: {tm}" 
            PETSc.Sys.Print(x)
            results[(ns, lev, cfl)] = (its, tm)

if MPI.COMM_WORLD.rank == 0:
    print(results)
    import pickle
    with open("stokesgmres.dat", "wb") as f:
        pickle.dump(results, f)
