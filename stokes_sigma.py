# Code used to estimate contraction constant for MG iteration
# for multi-stage heat equation
import gc
from copy import deepcopy

import numpy
from firedrake import (PCG64, Constant, DirichletBC,
                       DistributedMeshOverlapType, Function,
                       FunctionSpace, MeshHierarchy, RandomGenerator,
                       TestFunctions, UnitCubeMesh, curl, dx, inner, prolong)
from firedrake.petsc import PETSc
from irksome import Dt, RadauIIA, TimeStepper
from irksome.tools import IA
from mpi4py import MPI

dist_params = {"partition": True,
               "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}

rg = RandomGenerator(PCG64(seed=123456789))


params = {
    "snes_type": "ksponly",
    "ksp_type": "richardson",
    "ksp_richardson_scale": 1.0,
    "ksp_monitor": None,
    "ksp_gmres_restart": 100,
    "ksp_rtol": 1.e-12,
    "pc_type": "mg",
    "mg_levels": {
        "ksp_type": "chebyshev",
        "ksp_chebyshev_esteig": "0.0,0.25,0,1.2",
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


def get_random(mh):
    V0 = VectorFunctionSpace(mh[-1], "Q", 2)
    V1 = FunctionSpace(mh[-1], "DPC", 1)
    return rg.random(V0*V1)


def sigma(N_base, levels, cfl, deg, bt, tols):
    gc.collect()
    PETSc.Sys.Print(N_base, levels, cfl, deg, bt, tols)
    msh_base = UnitCubeMesh(N_base, N_base, N_base,
                            distribution_parameters=dist_params)
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

    t = Constant(0)
    dt = Constant(cfl / N_base / 2**levels)

    F = (inner(Dt(u), v) * dx
         + inner(dot(u, grad(u)), v) * dx
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
                 "ksp_type": "gmres",
                 "ksp_monitor": None,
                 "ksp_converged_reason": None,
                 "ksp_gmres_restart": 50,
                 "ksp_rtol": tols[0],
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

    stepper = TimeStepper(F, bt, t, dt, AA,
                          splitting=IA,
                          solver_parameters=it_params)

    ksp = stepper.solver.snes.getKSP()
    for myeps in tols:
        ksp.setTolerances(rtol=myeps)
        stepper.advance()
        
        its[myeps] = ksp.getIterationNumber()

    # estimate convergence rate:
    # sigma^(its) = eps
    # its log(sigma) = log(eps)
    # plot log(eps) vs its,
    # slope of line is (about) log(sigma)
    iterations = numpy.array([its[e] for e in tols])

    fit = numpy.polyfit(iterations,
                        numpy.log(tols),
                        1)
    PETSc.Sys.Print(fit)
    PETSc.Sys.Print(f"Estimated convergence rate: {numpy.exp(fit[0])}")
    return numpy.exp(fit[0])


N_base = 4
levels = 3

stages = (1, 2, 3, 4, 5)
tols = [1.e-4, 1.e-6, 1.e-8, 1.e-10]
degs = (1, 2)
cfls = (1, 8)

results = {}

for deg in degs:
    for ns in stages:
        bt = RadauIIA(ns)
        for cfl in cfls:
            sig = sigma(N_base, levels, cfl, deg, bt, tols)
            x = f"Degree {deg}, RadauIIA({ns}), cfl {cfl}: Sigma: {sig}"
            PETSc.Sys.Print(x)
            results[(deg, ns, cfl)] = sig

if MPI.COMM_WORLD.rank == 0:
    print(results)
    
    with open("eddysigma.csv", "w") as f:
        headers = [f"{deg}:{cfl}" for deg in degs for cfl in cfls]
        f.write(f"NS,{','.join(headers)}\n")
        for ns in stages:
            vals = [str(results[(deg, ns, cfl)]) for deg in degs for cfl in cfls]
            f.write(f"{ns},{','.join(vals)}\n")
