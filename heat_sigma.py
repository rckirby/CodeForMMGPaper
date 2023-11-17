# Code used to estimate contraction constant for MG iteration
# for multi-stage heat equation
# Appears in Figure 5.2 of final version
from copy import deepcopy

import numpy
from firedrake import (PCG64, Constant, DistributedMeshOverlapType, Function,
                       FunctionSpace, MeshHierarchy, RandomGenerator,
                       TestFunction, UnitCubeMesh, dx, grad, inner, prolong)
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
    Vs = [FunctionSpace(m, "CG", deg) for m in mh]
    noise = [rg.uniform(V, -1, 1) for V in Vs]

    for i in range(1, len(noise)):
        foo = Function(Vs[i])
        prolong(noise[i-1], foo)
        noise[i].assign(10 * foo + noise[i])

    return noise[-1]


def sigma(N_base, levels, cfl, deg, bt, tols):
    PETSc.Sys.Print(N_base, levels, cfl, deg, bt, tols)
    msh_base = UnitCubeMesh(N_base, N_base, N_base,
                            distribution_parameters=dist_params)
    mh = MeshHierarchy(msh_base, levels)
    msh = mh[-1]

    V = FunctionSpace(msh, "CG", deg)

    AA = get_random(mh, deg)
    v = TestFunction(V)

    t = Constant(0)
    dt = Constant(cfl / N_base / 2**levels)

    F = inner(Dt(AA), v) * dx + inner(grad(AA), grad(v)) * dx

    its = {}

    # measure iterations to convergence for each tolerance
    for myeps in tols:
        par_cur = deepcopy(params)
        par_cur["ksp_rtol"] = myeps

        stepper = TimeStepper(F, bt, t, dt, AA,
                              splitting=IA,
                              solver_parameters=par_cur)

        stepper.advance()

        its[myeps] = stepper.solver.snes.getKSP().getIterationNumber()

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
    
    with open("heatsigma.csv", "w") as f:
        headers = [f"{deg}:{cfl}" for deg in degs for cfl in cfls]
        f.write(f"NS,{','.join(headers)}\n")
        for ns in stages:
            vals = [str(results[(deg, ns, cfl)]) for deg in degs for cfl in cfls]
            f.write(f"{ns},{','.join(vals)}\n")
