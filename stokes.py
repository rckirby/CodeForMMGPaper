# Code used to generate date used in Figure 5.6
import petsc4py.PETSc
from firedrake import (Constant, DirichletBC, DistributedMeshOverlapType,
                       Function, FunctionSpace, Mesh, MeshHierarchy,
                       SpatialCoordinate, TestFunctions, VectorFunctionSpace,
                       as_vector, div, dot, dx, grad, inner, split)
from firedrake.petsc import PETSc
from irksome import Dt, RadauIIA, TimeStepper
from irksome.tools import IA
from mpi4py import MPI

# petsc4py.PETSc.Sys.popErrorHandler()

# PETSc.Log().begin()


# def get_time(event, comm=MPI.COMM_WORLD):
#     return comm.allreduce(PETSc.Log.Event(event).getPerfInfo()["time"],
#                           op=MPI.SUM) / comm.size


def run(levels, cfl, bt):
    dist_params = {"partition": True,
                   "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}

    msh = Mesh("obst.quad.msh", distribution_parameters=dist_params)
    mh = MeshHierarchy(msh, levels)
    msh = mh[-1]

    V = VectorFunctionSpace(msh, "Q", 2)
    W = FunctionSpace(msh, "DPC", 1)
    Z = V * W

    v, w = TestFunctions(Z)

    x, y = SpatialCoordinate(msh)

    up = Function(Z)
    u, p = split(up)

    dt = Constant(cfl * 0.1 / 2**levels, domain=msh)
    t = Constant(0, domain=msh)

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
    it_params = {
        "snes_type": "ksponly",
        "ksp_type": "gmres",
        "ksp_monitor": None,
        "ksp_gmres_restart": 50,
        "ksp_rtol": 1.e-8,
        "pc_type": "mg",
        "mg_levels": {
            "ksp_type": "chebyshev",
            # "ksp_chebyshev_esteig": "0.0,0.25,0,1.2",
            "ksp_max_it": 2,
            "ksp_convergence_test": "skip",
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMVankaPC",
            "pc_vanka_construct_codim": 0,
            "pc_vanka_exclude_subspaces": exclusions},
        "mg_coarse": {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 200}
    }

    stepper = TimeStepper(F, bt, t, dt, up,
                          bcs=bcs, solver_parameters=it_params,
                          splitting=IA)

    myksp = stepper.solver.snes.getKSP()
    for _ in range(3):
        stepper.advance()
        t.assign(float(t) + float(dt))

    stepper.advance()
    # sn = f"{levels}{cfl}{str(bt)}"
    # PETSc.Sys.Print(sn)
    # with PETSc.Log.Stage(sn):
    #     stepper.advance()
    #     # snes = get_time("KSPSolve")

    nv = FunctionSpace(msh, "CG", 1).dim()
    return (nv, 0, myksp.getIterationNumber())


with open(f"stokes.{MPI.COMM_WORLD.size}procs.csv", "w") as f:
    f.write("level,nv,stages,cfl,time,its\n")
    for level in (2, 3, 4, 5)[:1]:
        for cfl in (4,):
            for k in (1, 2, 3, 4, 5)[:1]:
                PETSc.Sys.Print(f"RadauIIA({k}) on refinement level {level} with cfl {cfl}:")  # noqa
                nv, tm, its = run(level, cfl, RadauIIA(k))
                PETSc.Sys.Print(f"   {nv} vertices, {tm} seconds, {its} iterations")  # noqa
                f.write(f"{level},{nv},{k},{cfl},{tm},{its}\n")
