# Code used to generate date used in Figure 5.2
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

PETSc.Log().begin()


def get_time(event, comm=MPI.COMM_WORLD):
    return comm.allreduce(PETSc.Log.Event(event).getPerfInfo()["time"],
                          op=MPI.SUM) / comm.size


params = {
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "ksp_gmres_restart": 100,
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
    rg = RandomGenerator(PCG64(seed=123456789))
    Vcoarse = FunctionSpace(mh[0], "CG", deg)
    u_noise = rg.uniform(Vcoarse, -1, 1)

    for m in mh[1:]:
        u_coarse = u_noise
        Vcoarse = FunctionSpace(m, "CG", deg)
        u_noise = Function(Vcoarse)
        prolong(u_coarse, u_noise)
        u_noise.assign(10*u_noise + rg.uniform(Vcoarse, -1, 1))

    msh = mh[-1]
    V = FunctionSpace(msh, "CG", 1)

    u_coarse = u_noise
    u_noise = rg.uniform(V, -1, 1)
    u_noise.assign(u_noise+u_coarse)
    return u_noise


def run(N_base, levels, cfl, deg, bt):
    msh_base = UnitCubeMesh(N_base, N_base, N_base,
                            distribution_parameters=dist_params)
    mh = MeshHierarchy(msh_base, levels)
    msh = mh[-1]

    V = FunctionSpace(msh, "CG", deg)

    AA = get_random(mh, deg)
    v = TestFunction(V)

    t = Constant(0, domain=msh)
    dt = Constant(cfl / N_base / 2**levels)

    F = inner(Dt(AA), v) * dx + inner(grad(AA), grad(v)) * dx

    stepper = TimeStepper(F, bt, t, dt, AA,
                          splitting=IA,
                          solver_parameters=params)

    myksp = stepper.solver.snes.getKSP()
    stepper.advance()
    sn = f"{N_base}{levels}{cfl}{deg}{str(bt)}{cfl}"
    PETSc.Sys.Print(sn)
    with PETSc.Log.Stage(sn):
        stepper.advance()
        snes = get_time("KSPSolve")

    nv = FunctionSpace(msh, "CG", 1).dim()
    return (nv, snes, myksp.getIterationNumber())


N_base = 4

with open(f"heat.{MPI.COMM_WORLD.size}procs.foo.csv", "w") as f:
    f.write("deg,level,nv,stages,cfl,time,its\n")
    for deg in (1, 2):
        for levels in (1, 2, 3)[:2]:
            for cfl in (1, 4, 8):
                for k in (1, 2, 3, 4, 5)[:2]:
                    PETSc.Sys.Print(f"RadauIIA({k}) on refinement level {levels} for degree {deg} with cfl {cfl}:")  # noqa
                    nv, tm, its = run(N_base, levels, cfl, deg, RadauIIA(k))
                    PETSc.Sys.Print(f"   {nv} vertices, {tm} seconds, {its} iterations")  # noqa
                    f.write(f"{deg},{levels},{nv},{k},{cfl},{tm},{its}\n")
