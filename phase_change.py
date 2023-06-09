import taichi as ti
# from CGSolver import CGSolver
from Pressure_CGSolver import Pressure_CGSolver
from Temperature_CGSolver import Temperature_CGSolver
from MICPCGSolver import MICPCGSolver
# from MGPCGSolver import MGPCGSolver
from Pressure_MGPCGSolver import Pressure_MGPCGSolver
from Temperature_MGPCGSolver import Temperature_MGPCGSolver
import numpy as np
from utils import ColorMap, vec2, vec3, clamp
import utils
import time
import math

debug = True
# ti.init(arch=ti.gpu, default_fp=ti.f32)
ti.init(arch = 'cpu', default_fp=ti.f32, debug=debug)

# ---------------params in simulation----------------
cell_res = 256
npar = 2

m = cell_res
n = cell_res
w = 10
h = 10 * n / m
grid_x = w / m
grid_y = h / n
inv_grid_x = 1.0 / grid_x
inv_grid_y = 1.0 / grid_y
pspace_x = grid_x / npar
pspace_y = grid_y / npar

p_vol, p_rho = (grid_x * 0.5)**2, 1 # vol: m^2, rho: kg/m^2
p_mass = p_vol * p_rho # mass: kg
g = -9.8
substeps = 4

# mass_min_thres = 1e-4

# Young's modulas , Poisson's ratio and lame parameter
# E_solid_phase, nu_solid_phase = 9, 0.3  # Young's modulus(Gpa) and Poisson's ratio for ice 
E_solid_phase, nu_solid_phase = 9 * 1e-4, 0.3  # Young's modulus(Gpa) and Poisson's ratio for ice 
mu_solid_phase_init, lambda_solid_phase_init = E_solid_phase / (2 * (1 + nu_solid_phase)), E_solid_phase * nu_solid_phase / (
    (1 + nu_solid_phase) * (1 - 2 * nu_solid_phase))  # Lame parameters

# E_fluid_phase, nu_fluid_phase = 2, 0.45  # Young's modulus(Gpa) and Poisson's ratio for water
E_fluid_phase, nu_fluid_phase = 2 * 1e-4, 0.45  # Young's modulus(Gpa) and Poisson's ratio for water
mu_fluid_phase_init, lambda_fluid_phase_init = E_fluid_phase / (2 * (1 + nu_fluid_phase)), E_fluid_phase * nu_fluid_phase / (
    (1 + nu_fluid_phase) * (1 - 2 * nu_fluid_phase))  # Lame parameters

# Initial value
T_air_init = 60 # 60 C
T_solid_phase_init = 100 # 100 C
T_fluid_phase_init = 0

# heat capacity (J/kg*c)
c_solid_phase_init = 2093
c_fluid_phase_init = 4182

# heat conductivity(W/mK)
k_solid_phase_init = 2.18
k_fluid_phase_init = 0.606
k_air_init = 0.04

# phase change parameters
freezing_point = 0 # 273K
latent = 382000 # latent of ice，J/kg

# ----------------fields---------------
# face part
# velocity field
u = ti.field(dtype=ti.f32, shape=(m + 1, n))
v = ti.field(dtype=ti.f32, shape=(m, n + 1))
u_face_mass = ti.field(dtype=ti.f32, shape=(m + 1, n))
v_face_mass = ti.field(dtype=ti.f32, shape=(m, n + 1))

# internal force
fu = ti.field(dtype=ti.f32, shape=(m + 1, n))
fv = ti.field(dtype=ti.f32, shape=(m, n + 1))

# heat conductivity field
k_u = ti.field(dtype=ti.f32, shape=(m + 1, n))
k_v = ti.field(dtype=ti.f32, shape=(m, n + 1))

# volume
vol_u = ti.field(dtype=ti.f32, shape=(m + 1, n))
vol_v = ti.field(dtype=ti.f32, shape=(m, n + 1))

# cell part
cell_mass = ti.field(dtype=ti.f32, shape=(m, n))
# pressure field
p = ti.field(dtype=ti.f32, shape=(m, n))

# J & Je & Jp
J = ti.field(dtype=ti.f32, shape=(m , n))
Je = ti.field(dtype=ti.f32, shape=(m , n))
Jp = ti.field(dtype=ti.f32, shape=(m , n))

# heat capacity
c = ti.field(dtype=ti.f32, shape=(m , n))

# temperature
T = ti.field(dtype=ti.f32, shape=(m , n))

# 1 / lambda
inv_lambda = ti.field(dtype=ti.f32, shape=(m , n))

# -------------particle properties------------
# particle x, y
particle_positions = ti.Vector.field(2, dtype=ti.f32, shape=(m, n, npar, npar))
particle_velocities = ti.Vector.field(2,
                                      dtype=ti.f32,
                                      shape=(m, n, npar, npar))
# particle physical parameter

# particle C
cp_x = ti.Vector.field(2, dtype=ti.f32, shape=(m, n, npar, npar))
cp_y = ti.Vector.field(2, dtype=ti.f32, shape=(m, n, npar, npar))

# particle F
particle_Fe = ti.Matrix.field(2, 2, dtype=ti.f32, shape=(m, n, npar, npar))  # Elastic deformation gradient (F_E)
particle_Fp = ti.Matrix.field(2, 2, dtype=ti.f32, shape=(m, n, npar, npar))  # Plastic deformation gradient (F_P)

# particle type
particle_type = ti.field(dtype=ti.i32, shape=(m, n, npar, npar))
P_FLUID = 1
P_OTHER = 0

# particle lame's parameter
particle_mu = ti.field(dtype=ti.f32, shape=(m, n, npar, npar))
particle_la = ti.field(dtype=ti.f32, shape=(m, n, npar, npar)) # lambda

# particle heat parameters
particle_T = ti.field(dtype=ti.f32, shape=(m, n, npar, npar)) # temperature
particle_last_T = ti.field(dtype=ti.f32, shape=(m, n, npar, npar)) # last temperature
particle_U = ti.field(dtype=ti.f32, shape=(m, n, npar, npar)) # heat
particle_c = ti.field(dtype=ti.f32, shape=(m, n, npar, npar)) # heat capacity
particle_k = ti.field(dtype=ti.f32, shape=(m, n, npar, npar)) # heat conductivity
particle_l = ti.field(dtype=ti.f32, shape=(m, n, npar, npar)) # latent heat
particle_Phase = ti.field(dtype=ti.i32, shape=(m, n, npar, npar))
P_FLUID_PHASE = 3
P_SOLID_PHASE = 4
P_OTHER_PHASE = 5

# -----------------params in render-----------
screen_res = (800, 800 * n // m)
bwrR = ColorMap(1.0, .25, 1, .5)
bwrG = ColorMap(1.0, .5, .5, .5)
bwrB = ColorMap(1.0, 1, .25, .5)
color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=screen_res)
gui = ti.GUI("watersim2D", screen_res, show_gui = False)

# cell type
cell_type = ti.field(dtype=ti.i32, shape=(m, n))


# pressure solver
# preconditioning = 'MG'
preconditioning = None

MIC_blending = 0.97

mg_level = 4
pre_and_post_smoothing = 2
bottom_smoothing = 10

dt = 0.01

if preconditioning == None:
    # solver = CGSolver(m, n, u, v, cell_type)
    pressure_solver = Pressure_CGSolver(m, n, u, v, dt, Jp, Je, inv_lambda, cell_type, vol_u, vol_v, u_face_mass, v_face_mass)
    temperature_solver = Temperature_CGSolver(m, n, k_u, k_v, T, c, cell_type, cell_mass, grid_x)
elif preconditioning == 'MIC':
    solver = MICPCGSolver(m, n, u, v, cell_type, MIC_blending=MIC_blending)
elif preconditioning == 'MG':
    pressure_solver = Pressure_MGPCGSolver(m, n, u, v, dt, Jp, Je, inv_lambda, cell_type, multigrid_level=mg_level,
                         pre_and_post_smoothing=pre_and_post_smoothing,
                         bottom_smoothing=bottom_smoothing)
    temperature_solver = Temperature_MGPCGSolver(m, n, k_u, k_v, T, c, cell_type, multigrid_level=mg_level,
                         pre_and_post_smoothing=pre_and_post_smoothing,
                         bottom_smoothing=bottom_smoothing)
    # temperature_solver = Temperature_CGSolver(m, n, k_u, k_v, T, c, cell_type)

# save to gif
result_dir = "./results"
video_manager = ti.tools.VideoManager(output_dir=result_dir,
                                framerate=24,
                                automatic_build=False)

# --------------------rendering------------------
def render():
    render_type = 'particles'

    @ti.func
    def map_color(c):
        return vec3(bwrR.map(c), bwrG.map(c), bwrB.map(c))

    @ti.kernel
    def fill_marker():
        for i, j in color_buffer:
            x = int((i + 0.5) / screen_res[0] * w / grid_x)
            y = int((j + 0.5) / screen_res[1] * h / grid_y)

            m = cell_type[x, y]
            # m = T[x, y] / 373

            color_buffer[i, j] = map_color(m * 0.5)

    def render_pixels():
        fill_marker()
        img = color_buffer.to_numpy()
        gui.set_image(img) 

    def render_particles():
        bg_color = 0x112f41
        particle_color = 0x068587
        particle_radius = 1.0
        fluid_color = 0xC8D804
        solid_color = 0xC3ABCD

        pf = particle_type.to_numpy()
        np_type = pf.copy()
        np_type = np.reshape(np_type, -1)

        pp = particle_Phase.to_numpy()
        np_phase = pp.copy()
        np_phase = np.reshape(np_phase, -1)

        pos = particle_positions.to_numpy()
        np_pos = pos.copy()
        np_pos_fluid = pos.copy()
        np_pos_solid = pos.copy()
        np_pos = np.reshape(pos, (-1, 2))
        np_pos_fluid = np.reshape(pos, (-1, 2))
        np_pos_solid = np.reshape(pos, (-1, 2))
        np_pos = np_pos[np.where(np_type == P_FLUID)]
        np_pos_fluid = np_pos_fluid[np.where(np_phase == P_FLUID_PHASE)]
        np_pos_solid = np_pos_solid[np.where(np_phase == P_SOLID_PHASE)]
        for i in range(np_pos.shape[0]):
            np_pos[i][0] /= w
            np_pos[i][1] /= h
        for i in range(np_pos_fluid.shape[0]):
            np_pos_fluid[i][0] /= w
            np_pos_fluid[i][1] /= h
        for i in range(np_pos_solid.shape[0]):
            np_pos_solid[i][0] /= w
            np_pos_solid[i][1] /= h

        gui.clear(bg_color)
        # gui.circles(np_pos, radius=particle_radius, color=particle_color)
        gui.circles(np_pos_fluid, radius=particle_radius, color=fluid_color)
        gui.circles(np_pos_solid, radius=particle_radius, color=solid_color)
    
    if render_type == 'particles':
        render_particles()
    else:
        render_pixels()

    video_manager.write_frame(gui.get_image())

# ------------initialization-----------
def init():
    # init scene
    @ti.kernel
    def init_dambreak(x: ti.f32, y: ti.f32):
        xn = int(x / grid_x)
        yn = int(y / grid_y)

        for i, j in cell_type:
            if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                cell_type[i, j] = utils.SOLID  # boundary
            else:
                if i <= xn and j <= yn:
                    cell_type[i, j] = utils.FLUID
                else:
                    cell_type[i, j] = utils.AIR

    @ti.kernel
    def init_spherefall(xc: ti.f32, yc: ti.f32, r: ti.f32):
        for i, j in cell_type:
            if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                cell_type[i, j] = utils.SOLID  # boundary
            else:
                x = (i + 0.5) * grid_x
                y = (j + 0.5) * grid_y

                phi = (x - xc)**2 + (y - yc) ** 2 - r**2

                if phi <= 0 :
                    cell_type[i, j] = utils.FLUID
                else:
                    cell_type[i, j] = utils.AIR

    #init simulation
    @ti.kernel
    def init_field():
        for i, j in u:
            u[i, j] = 0.0

        for i, j in v:
            v[i, j] = 0.0

        for i, j in u_face_mass:
            u_face_mass[i, j] = -1.0
        
        for i, j in v_face_mass:
            v_face_mass[i, j] = -1.0

        for i, j in fu:
            fu[i, j] = 0.0
        
        for i, j in fv:
            fv[i, j] = 0.0

        for i, j in k_u:
            k_u[i, j] = 0

        for i, j in k_v:
            k_v[i, j] = 0

        for i, j in vol_u:
            vol_u[i, j] = 0.0
        
        for i, j in vol_v:
            vol_v[i, j] = 0.0

        for i, j in cell_mass:
            cell_mass[i, j] = -1.0

        for i, j in p:
            p[i, j] = 0.0


        for i, j in J:
            J[i, j] = 1
            Je[i, j] = 1
            Jp[i, j] = 1
            c[i, j] = 0
            T[i, j] = 0
            inv_lambda[i, j] = 0

    @ti.kernel
    def init_particles():
        for i, j, ix, jx in particle_positions:
            if cell_type[i, j] == utils.FLUID:
                particle_type[i, j, ix, jx] = P_FLUID
                particle_Phase[i, j, ix, jx] = P_SOLID_PHASE
            else:
                particle_type[i, j, ix, jx] = P_OTHER
                particle_type[i, j, ix, jx] = P_OTHER_PHASE
            px = i * grid_x + (ix + ti.random(ti.f32)) * pspace_x
            py = j * grid_y + (jx + ti.random(ti.f32)) * pspace_y

            particle_positions[i, j, ix, jx] = vec2(px, py)
            particle_velocities[i, j, ix, jx] = vec2(0.0, 0.0)
            cp_x[i, j, ix, jx] = vec2(0.0, 0.0)
            cp_y[i, j, ix, jx] = vec2(0.0, 0.0)
            particle_Fe[i, j, ix, jx] = ti.Matrix.identity(dt = ti.f32, n=2)
            particle_Fp[i, j, ix, jx] = ti.Matrix.identity(dt = ti.f32, n=2)
            particle_mu[i, j, ix, jx] = mu_solid_phase_init
            particle_la[i, j, ix, jx] = lambda_solid_phase_init
            particle_T[i, j, ix, jx] = T_fluid_phase_init
            particle_last_T[i, j, ix, jx] = T_fluid_phase_init
            particle_U[i, j, ix, jx] = 0  # [0, Lp], initialilly at 0
            particle_c[i, j, ix, jx] = c_solid_phase_init # for ice J/kg * C
            particle_k[i, j, ix, jx] = k_solid_phase_init
            particle_l[i, j, ix, jx] = p_mass * latent # for ice J/kg to melt            


    # init_dambreak(4, 4)
    # init_spherefall(5,3,2)
    init_spherefall(5, 2, 2)
    init_field()
    init_particles()

# -------------subprocesses----------------

            
            

@ti.kernel
def deformation_gradient_add_plasticity():
    for p in ti.grouped(particle_positions):
        if particle_type[p] == P_FLUID:
            U, sig, V = ti.svd(particle_Fe[p])
            theta_c = 2.5e-2
            theta_s = 4.5e-3
            for d in ti.static(range(2)):
                sig[d, d] = ti.min(ti.max(sig[d, d], 1 - theta_c), 1 + theta_s)  # Plasticity
            Jp = particle_Fp[p].determinant()
            # Note: this part is (Jp)^(1/d), here d = 2
            particle_Fe[p] = ti.sqrt(Jp) * U@sig@V
            particle_Fp[p] = (1 / ti.sqrt(Jp)) * particle_Fp[p]

@ti.func
def cubic_bspline_kernal(fx):
    w = [vec2(1.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 1.0)]
    if fx[0] > 1.0:
        w[0][0] *= ( (-1.0 / 6.0) * (fx[0] ** 3) + (fx[0] ** 2) - 2 * fx[0] + (4.0 / 3.0))
    else:
        w[0][0] *= ((1.0 / 2.0) * (fx[0] ** 3) - (fx[0] ** 2) + (2.0 / 3.0))
    if fx[1] > 1.0:
        w[0][1] *= ( (-1.0 / 6.0) * (fx[1] ** 3) + (fx[1] ** 2) - 2 * fx[1] + (4.0 / 3.0))
    else:
        w[0][1] *= ((1.0 / 2.0) * (fx[1] ** 3) - (fx[1] ** 2) + (2.0 / 3.0))
    
    w[1] = ((1.0 / 2.0) * (ti.abs(fx-1) ** 3) - (ti.abs(fx-1) ** 2) + (2.0 / 3.0))
    
    if fx[0]-2 < -1.0:
        w[2][0] *= ( (-1.0 / 6.0) * (ti.abs(fx[0]-2) ** 3) + ((fx[0]-2) ** 2) - 2 * ti.abs(fx[0]-2) + (4.0 / 3.0))
    else:
        w[2][0] *= ((1.0 / 2.0) * (ti.abs(fx[0]-2) ** 3) - ((fx[0]-2) ** 2) + (2.0 / 3.0))
    if fx[1]-2 < -1.0:
        w[2][1] *= ( (-1.0 / 6.0) * (ti.abs(fx[1]-2) ** 3) + ((fx[1]-2) ** 2) - 2 * ti.abs(fx[1]-2) + (4.0 / 3.0))
    else:
        w[2][1] *= ((1.0 / 2.0) * (ti.abs(fx[1]-2) ** 3) - ((fx[1]-2) ** 2) + (2.0 / 3.0))
    return w

@ti.func
def cubic_bspline_kernal_grad(fx):
    return [(-1.0/2.0) * (fx ** 2) + 2 * fx - 2, (3.0 / 2.0) * ((fx-1) ** 2) - (2 * (fx-1)), (-1.0/2.0) * ((fx-2) ** 2) + 2 * (fx-2) - 2]

@ti.func
def scatter_face_u(xp, vp, cp, PFT, kp):
    stagger = vec2(0.0, 0.5)
    e = vec2(1.0, 0.0)
    inv_dx = vec2(inv_grid_x, inv_grid_y).cast(ti.f32)
    # inv_dx = vec2(1.0 / grid_x, 1.0 / grid_y).cast(ti.f32)
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)
    # print(xp)
    # Note, in SSJCTS14, they use cubic B-spline
    w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Quadratic Bspline

    # cubic spline

    # w_cdf = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]
    w_grad = [fx-1.5, -2*(fx-1), fx-3.5] # Bspline gradient
    # w = cubic_bspline_kernal(fx)
    # w_grad = cubic_bspline_kernal_grad(fx)
    # print("vp: ", vp)
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = vec2(i, j)
            if vec2_is_valid(base + offset):
                dpos = (offset.cast(ti.f32) - fx) * vec2(grid_x, grid_y)
                # print(w)
                weight_grad = vec2(w_grad[i][0]*w[j][1], w[i][0]*w_grad[j][1])
                weight = w[i][0] * w[j][1] # x, y directions, respectively
                # weight_cdf = w_cdf[i] * w_cdf[j]

                # su = base + offset
                u[base + offset] += weight * p_mass * vp#(vp + ti.math.dot(cp, dpos))
                k_u[base + offset] += weight * p_mass * kp   # Need not to multiply affine to heat conductivity(maybe?)
                if u_face_mass[base + offset] < 0:
                    u_face_mass[base + offset] = weight * p_mass # Maybe in waterSim2d, they assume that every particles' mass is 1?
                else:
                    u_face_mass[base + offset] += weight * p_mass # Maybe in waterSim2d, they assume that every particles' mass is 1?
                fu[base + offset] += ti.math.dot(e, (-PFT) @ weight_grad )
                # vol_u[base + offset] += weight_cdf

@ti.func
def scatter_face_v(xp, vp, cp, PFT, kp):
    stagger = vec2(0.5, 0.0)
    e = vec2(0.0, 1.0)
    inv_dx = vec2(inv_grid_x, inv_grid_y).cast(ti.f32)
    # inv_dx = vec2(1.0 / grid_x, 1.0 / grid_y).cast(ti.f32)
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)
    # print(xp)
    # Note, in SSJCTS14, they use cubic B-splinew = cubic_bspline_kernal(fx)
    # w = cubic_bspline_kernal(fx)
    # w_grad = cubic_bspline_kernal_grad(fx)
    w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Quadratic Bspline
    # w_cdf = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]
    w_grad = [fx-1.5, -2*(fx-1), fx-3.5] # Bspline gradient
    # print("vp: ", vp)
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = vec2(i, j)
            if vec2_is_valid(base + offset):
                dpos = (offset.cast(ti.f32) - fx) * vec2(grid_x, grid_y)
                weight_grad = vec2(w_grad[i][0]*w[j][1], w[i][0]*w_grad[j][1])
                weight = w[i][0] * w[j][1] # x, y directions, respectively
                # weight_cdf = w_cdf[i] * w_cdf[j]

                # print("v" + str(i) + ", " + str(j) + ": ", weight * p_mass * (vp + cp.dot(dpos)))
                v[base + offset] += weight * p_mass * vp#(vp + ti.math.dot(cp, dpos))
                k_v[base + offset] += weight * p_mass * kp   # Need not to multiply affine to heat conductivity(maybe?)
                if v_face_mass[base + offset] < 0.0:
                    v_face_mass[base + offset] = weight * p_mass # Maybe in waterSim2d, they assume that every particles' mass is 1?
                else:
                    v_face_mass[base + offset] += weight * p_mass # Maybe in waterSim2d, they assume that every particles' mass is 1?
                fv[base + offset] += ti.math.dot(e, (-PFT) @ weight_grad )
                # vol_v[base + offset] += weight_cdf

@ti.func
def scatter_cell(xp, par_J, par_Je, par_c, par_T, par_inv_lambda):
    inv_dx = vec2(inv_grid_x, inv_grid_y).cast(ti.f32)
    base = (xp * inv_dx - 0.5).cast(ti.i32)
    fx = xp * inv_dx - base.cast(ti.f32)
    # Note, in SSJCTS14, they use cubic B-splines
    w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2] # Quadratic Bspline
    # w = cubic_bspline_kernal(fx)
    # inv_dx = vec2(1.0 / grid_x, 1.0 / grid_y).cast(ti.f32)
    # base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    # fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = vec2(i, j)
            if vec2_is_valid(base + offset):
                # dpos = (offset.cast(ti.f32) - fx) * vec2(grid_x, grid_y)
                weight = w[i][0] * w[j][1] # x, y directions, respectively
                if cell_mass[i, j] < 0.0:
                    cell_mass[base + offset] = weight * p_mass
                else:
                    cell_mass[base + offset] += weight * p_mass
                Je[base + offset] += weight * p_mass * par_Je
                J[base + offset] += weight * p_mass * par_J
                c[base + offset] += weight * p_mass * par_c
                T[base + offset] += weight * p_mass * par_T
                inv_lambda[base + offset] += weight * p_mass * par_inv_lambda
                # print("index:", i, j)
                # print("weight: ", weight)
                """
                su = base + offset
                if Je[su] < 1e-10:
                    print("index: ", su[0], su[1])
                    print("Je: ", Je[su])
                    print("pmass: ", p_mass)
                    print("weight: ", weight)
                """

@ti.func
def set_Jp():
    for i, j in Jp:
        if Je[i, j] > 0.0:
            Jp[i, j] = J[i, j] / Je[i, j]


@ti.kernel
def P2G():
    # stagger_u = vec2(0.0, 0.5)
    # stagger_v = vec2(0.5, 0.0)
    for p in ti.grouped(particle_positions):
        if particle_type[p] == P_FLUID:
            xp = particle_positions[p]
            # par_F = particle_Fe[p] @ particle_Fp[p]
            par_Fe = particle_Fe[p]
            par_Je = par_Fe.determinant()
            par_Jp = particle_Fp[p].determinant()
            """
            if par_Je < 0.0 or par_Jp < 0.0:
                print("par_Je: ", par_Je)
                print("Fe: ", particle_Fe[p])
                print("par_Jp: ", par_Jp)
                print("Fp: ", particle_Fp[p])
            """
            par_J = par_Je * par_Jp
            U, sig, V = ti.svd(par_Fe)
            # P(F) F^T
            par_f = 2 * particle_mu[p] * (par_Fe - U@V.transpose()) @ par_Fe.transpose() + ti.Matrix.identity(ti.f32, 2) * particle_la[p] * par_J * (par_J-1)
            par_f = p_vol * par_f # f is 2 by 2 matrix, which will be multiplid by weight gradient
            # face
            scatter_face_u(xp, particle_velocities[p][0], cp_x[p], par_f, particle_k[p])
            scatter_face_v(xp, particle_velocities[p][1], cp_y[p], par_f, particle_k[p])
            # scatter_face(u, u_face_mass, k_u, fu, xp, particle_velocities[p][0],
              #               cp_x[p], par_f, stagger_u, particle_k[p], vec2(1.0, 0.0))
            # scatter_face(v, v_face_mass, k_v, fv, xp, particle_velocities[p][1],
              #               cp_y[p], par_f, stagger_v, particle_k[p], vec2(0.0, 1.0))
            # cell
            # par_Je = particle_Fe[p].determinant()
            # par_Jp = particle_Fp[p].determinant()
            # par_J = par_Je * par_Jp
            par_c = particle_c[p]
            par_T = particle_T[p]
            par_inv_lambda = 1.0 / particle_la[p]
            scatter_cell(xp, par_J, par_Je, par_c, par_T, par_inv_lambda)
    set_Jp() # Jp = J / Je for all cell
@ti.kernel
def clear_field():
    u.fill(0.0)
    v.fill(0.0)
    u_face_mass.fill(-1.0)
    v_face_mass.fill(-1.0)
    fu.fill(0.0)
    fv.fill(0.0)
    k_u.fill(0.0)
    k_v.fill(0.0)
    vol_u.fill(0.0)
    vol_v.fill(0.0)
    cell_mass.fill(-1.0)
    p.fill(0.0)
    J.fill(0.0)
    Je.fill(0.0)
    Jp.fill(0.0)
    c.fill(0.0)
    T.fill(0.0)
    inv_lambda.fill(0.0)


@ti.func
def is_valid(i, j):
    return i >= 0 and i < m and j >= 0 and j < n

@ti.func
def vec2_is_valid(v):
    return is_valid(v[0], v[1])

@ti.func
def is_fluid(i, j):
    return is_valid(i, j) and cell_type[i, j] == utils.FLUID


@ti.func
def is_solid(i, j):
    # return is_valid(i, j) and cell_type[i, j] == utils.SOLID
    return i == 0 or i == m - 1 or j == 0 or j == n - 1


@ti.func
def is_air(i, j):
    return is_valid(i, j) and cell_type[i, j] == utils.AIR


@ti.kernel
def mark_cell():
    # AIR: empty
    # SOLID: colliding
    # FLUID: interior
    # mark cells corresponding to SSJCTS14
    for i, j in cell_type:
        if not is_solid(i, j):
            # check if it's interior i.e. every face has mass
            # for cell(i, j), faces are u(i, j), u(i+1, j), v(i, j), v(i, j+1)
            # print(u_face_mass[i, j])
            # print(u_face_mass[i+1, j])
            # print(v_face_mass[i, j])
            # print(v_face_mass[i, j+1])
            if u_face_mass[i, j] > 0.0 and u_face_mass[i+1, j] > 0.0 and v_face_mass[i, j] > 0.0 and v_face_mass[i, j+1] > 0.0 and cell_mass[i, j] > 0.0:
                cell_type[i, j] = utils.FLUID
            else:
                cell_type[i, j] = utils.AIR
@ti.kernel
def assign_temperature():
    for i, j in cell_type:
        if cell_type == utils.AIR:
            # air: 60 C
            T[i, j] = T_air_init
        elif cell_type == utils.SOLID:
            # solid(edge): 100 C
            T[i, j] = T_solid_phase_init
        


@ti.kernel
def face_normalize():
    for i, j in u:
        if u_face_mass[i, j] > 0.0:
            u[i, j] = u[i, j] / u_face_mass[i, j]

    for i, j in v:
        # pr = False
        # if v[i, j] > 1e-2:
            # print("Before v: ", v[i, j], " index: ", i, j)
            # print("face_mass: ", v_face_mass[i, j])
         #    pr = True
        if v_face_mass[i, j] > 0.0:
            v[i, j] = v[i, j] / v_face_mass[i, j]
        # if pr:        
            # print("After v: ", v[i, j], " index: ", i, j)

    for i, j in k_u:
        if u_face_mass[i, j] > 0.0:
            k_u[i, j] = k_u[i, j] / u_face_mass[i, j]

    for i, j in k_v:
        if v_face_mass[i, j] > 0.0:
            k_v[i, j] = k_v[i, j] / v_face_mass[i, j]

@ti.kernel
def cell_normalize():
    for i, j in J:
        if cell_mass[i, j] > 0.0:
            J[i, j] /= cell_mass[i, j]

    for i, j in Je:
        if cell_mass[i, j] > 0.0:
            Je[i, j] /= cell_mass[i, j]

    for i, j in c:
        if cell_mass[i, j] > 0.0:
            c[i, j] /= cell_mass[i, j]

    for i, j in T:
        # print("Before temp: ", T[i, j], ", index: ", i, j)
        # print("cell mass: ", cell_mass[i, j])
        if cell_mass[i, j] > 0.0:
            T[i, j] /= cell_mass[i, j]
        # print("After temp: ", T[i, j], ", index: ", i, j)

    for i, j in inv_lambda:
        if cell_mass[i, j] > 0.0:
            inv_lambda[i, j] /= cell_mass[i, j]
    
@ti.kernel
def apply_force(dt: ti.f32):

    # internal force (have been calculated in P2G)
    for i, j in u:
        if u_face_mass[i, j] > 0.0:
            u[i, j] += (fu[i, j]/u_face_mass[i, j]) * dt
    
    for i, j in v:
        if v_face_mass[i, j] > 0.0:
            v[i, j] += (fv[i, j]/v_face_mass[i, j]) * dt

    # gravity(only v(y) direction)
    for i, j in v:
        v[i, j] += g * dt
        
@ti.kernel
def enforce_boundary():
    # u solid
    for i, j in u:
        if is_solid(i - 1, j) or is_solid(i, j):
            u[i, j] = 0.0
            # u[i, j] = -u[i, j]        

    # v solid
    for i, j in v:
        if is_solid(i, j - 1) or is_solid(i, j):
            v[i, j] = 0.0
            # v[i, j] = -v[i, j]

@ti.kernel
def collect_face_vol():
    # cubic b spline cdf
    w_4 = [0.041667, 0.45833, 0.45833, 0.041667]
    w_5 = [0.0026042, 0.1979125, 0.59896, 0.1979125, 0.0026042] 
    for i, j in vol_u:
        for offsetX in ti.static(range(4)):
            for offsetY in ti.static(range(5)):
                if is_fluid(i-2+offsetX, j-2+offsetY):
                    vol_u[i, j] += w_4[offsetX] * w_5[offsetY]

    for i, j in vol_v:
        for offsetX in ti.static(range(5)):
            for offsetY in ti.static(range(4)):
                if is_fluid(i-2+offsetX, j-2+offsetY):
                    vol_v[i, j] += w_5[offsetX] * w_4[offsetY]

def solve_pressure(dt: ti.f32):
    scale_A = dt / (grid_x * grid_x)
    scale_b = 1 / grid_x

    """
    pressure_solver.u.copy_from(u)
    pressure_solver.v.copy_from(v)
    pressure_solver.Jp.copy_from(Jp)
    pressure_solver.Je.copy_from(Je)
    pressure_solver.inv_lambda.copy_from(inv_lambda)
    pressure_solver.cell_type.copy_from(cell_type)
    """

    pressure_solver.system_init(scale_A, scale_b)
    pressure_solver.solve(500)

    p.copy_from(pressure_solver.p)

@ti.kernel
def apply_pressure(dt: ti.f32):
    # scale = dt / (p_rho * grid_x)
    scale = dt / grid_x

    for i, j in ti.ndrange(m, n):
        if is_fluid(i - 1, j) or is_fluid(i, j):
            if is_solid(i - 1, j) or is_solid(i, j):
                u[i, j] = 0
            else:
                inv_rho = vol_u[i, j] / u_face_mass[i, j]
                u[i, j] += scale * (p[i, j] - p[i - 1, j]) * inv_rho
                """
                if isnan(u[i, j]):
                    print("apply pressure to let u be nan")
                    print("u[i, j]: ", u[i, j])
                    print("pressure: ", p[i, j], p[i-1, j])
                    print("index: ", i, j)
                """

        if is_fluid(i, j - 1) or is_fluid(i, j):
            if is_solid(i, j - 1) or is_solid(i, j):
                v[i, j] = 0
            else:
                inv_rho = vol_v[i, j] / v_face_mass[i, j]
                v[i, j] += scale * (p[i, j] - p[i, j - 1]) * inv_rho
                """
                if isnan(v[i, j]):
                    print("apply pressure to let v be nan")
                    print("v[i, j]: ", v[i, j])
                    print("pressure: ", p[i, j], p[i-1, j])
                    print("index: ", i, j)
                """

def solve_temperature(dt: ti.f32):
    scale_A = dt / (grid_x * grid_x)
    # scale_b = 1 / grid_x
    scale_b = 1

    """
    temperature_solver.k_u.copy_from(k_u)
    temperature_solver.k_v.copy_from(k_v)
    temperature_solver.old_T.copy_from(T)
    temperature_solver.c.copy_from(c)
    temperature_solver.cell_type.copy_from(cell_type)
    """
    temperature_solver.system_init(scale_A, scale_b)
    temperature_solver.solve(500)

    T.copy_from(temperature_solver.p)

@ti.func
def tight_quadra_kernal(fx):
    return [(1.0 / 2.0) * (fx ** 2) - (3.0 / 2.0) * fx + (9.0 / 8.0), -((fx-1)**2) + (3.0/4.0), (1.0 / 2.0) * ((fx-2) ** 2) - (3.0 / 2.0) * (fx-2) + (9.0 / 8.0)]

@ti.func
def tight_quadra_kernal_grad(fx):
    return [fx-(3.0/2.0), -2*(fx-1), (fx-2)-(3.0/2.0)]

@ti.func
def gather_vp_u(xp):
    stagger = vec2(0.0, 0.5)
    inv_dx = vec2(1.0 / grid_x, 1.0 / grid_y).cast(ti.f32)
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

    # w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Bspline
    # tight quadratic stencil
    w = tight_quadra_kernal(fx)

    v_pic = 0.0

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = vec2(i, j)
            if vec2_is_valid(base + offset):
                weight = w[i][0] * w[j][1]
                v_pic += weight * u[base + offset]
                # print(weight * grid_v[base + offset])
                """
                if isnan(v_pic):
                    print("u_pic is nan")
                    print("u_pic: ", v_pic)
                    print("index: ", i, j)
                    print("weight: ", weight)
                    print("u: ", u[base + offset])
                    print("base: ", base)
                """
    return v_pic

@ti.func
def gather_vp_v(xp):
    stagger = vec2(0.5, 0.0)
    inv_dx = vec2(1.0 / grid_x, 1.0 / grid_y).cast(ti.f32)
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

    # w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Bspline
    # w = [(1.0 / 2.0) * (fx ** 2) - (3.0 / 2.0) * fx + (9.0 / 8.0), -((fx-1)**2) + (3.0/4.0), (1.0 / 2.0) * ((fx-2) ** 2) - (3.0 / 2.0) * (fx-2) + (9.0 / 8.0)]
    w = tight_quadra_kernal(fx)

    v_pic = 0.0

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = vec2(i, j)
            if vec2_is_valid(base + offset):
                weight = w[i][0] * w[j][1]
                v_pic += weight * v[base + offset]
                """
                if isnan(v_pic):
                    print("v_pic is nan")
                    print("v_pic: ", v_pic)
                    print("index: ", i, j)
                    print("weight: ", weight)
                    print("v: ", v[base + offset])
                    print("base: ", base)
                # print(weight * grid_v[base + offset])
                """

    return v_pic

@ti.func
def gather_cp_x(xp):
    stagger = vec2(0.0, 0.5)
    inv_dx = vec2(1.0 / grid_x, 1.0 / grid_y).cast(ti.f32)
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

    # w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Bspline
    # w = [(1.0 / 2.0) * (fx ** 2) - (3.0 / 2.0) * fx + (9.0 / 8.0), -((fx-1)**2) + (3.0/4.0), (1.0 / 2.0) * ((fx-2) ** 2) - (3.0 / 2.0) * (fx-2) + (9.0 / 8.0)]
    w = tight_quadra_kernal(fx)

    w_grad = tight_quadra_kernal_grad(fx)
    # w_grad = [fx-(3.0/2.0), -2*(fx-1), (fx-2)-(3.0/2.0)]
    # w_grad = [fx-1.5, -2*(fx-1), fx-3.5] # Bspline gradient

    cp = vec2(0.0, 0.0)

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = vec2(i, j)
            if vec2_is_valid(base + offset):
                # dpos = offset.cast(ti.f32) - fx
                # weight = w[i][0] * w[j][1]
                # cp += 4 * weight * dpos * grid_v[base + offset] * inv_dx[0]
                weight_grad = vec2(w_grad[i][0]*w[j][1], w[i][0]*w_grad[j][1])
                cp += weight_grad * u[base + offset]    

    return cp

@ti.func
def gather_cp_y(xp):
    stagger = vec2(0.5, 0.0)
    inv_dx = vec2(1.0 / grid_x, 1.0 / grid_y).cast(ti.f32)
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

    # w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Bspline
    # w_grad = [fx-1.5, -2*(fx-1), fx-3.5] # Bspline gradient
    # w = [(1.0 / 2.0) * (fx ** 2) - (3.0 / 2.0) * fx + (9.0 / 8.0), -((fx-1)**2) + (3.0/4.0), (1.0 / 2.0) * ((fx-2) ** 2) - (3.0 / 2.0) * (fx-2) + (9.0 / 8.0)]

    # w_grad = [fx-(3.0/2.0), -2*(fx-1), (fx-2)-(3.0/2.0)]
    w = tight_quadra_kernal(fx)
    w_grad = tight_quadra_kernal_grad(fx)

    cp = vec2(0.0, 0.0)

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = vec2(i, j)
            if vec2_is_valid(base + offset):
                # dpos = offset.cast(ti.f32) - fx
                # weight = w[i][0] * w[j][1]
                # cp += 4 * weight * dpos * grid_v[base + offset] * inv_dx[0]
                weight_grad = vec2(w_grad[i][0]*w[j][1], w[i][0]*w_grad[j][1])
                cp += weight_grad * v[base + offset]

    return cp

@ti.func
def gather_Tp(xp):
    inv_dx = vec2(inv_grid_x, inv_grid_y).cast(ti.f32)
    base = (xp * inv_dx - 0.5).cast(ti.i32)
    fx = xp * inv_dx - base.cast(ti.f32)

    # w = [(1.0 / 2.0) * (fx ** 2) - (3.0 / 2.0) * fx + (9.0 / 8.0), -((fx-1)**2) + (3.0/4.0), (1.0 / 2.0) * ((fx-2) ** 2) - (3.0 / 2.0) * (fx-2) + (9.0 / 8.0)]
    w = tight_quadra_kernal(fx)

    Tp = 0.0

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = vec2(i, j)
            if vec2_is_valid(base + offset):
                # dpos = offset.cast(ti.f32) - fx
                weight = w[i][0] * w[j][1]
                # cp += 4 * weight * dpos * grid_v[base + offset] * inv_dx[0]
                Tp += weight * T[base + offset]

    return Tp

@ti.func
def gather_vp_grad_u(xp):
    stagger = vec2(0.0, 0.5)
    e = vec2(1.0, 0.0)
    inv_dx = vec2(1.0 / grid_x, 1.0 / grid_y).cast(ti.f32)
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

    # w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Bspline
    # w_grad = [fx-1.5, -2*(fx-1), fx-3.5] # Bspline gradient
    # w = [(1.0 / 2.0) * (fx ** 2) - (3.0 / 2.0) * fx + (9.0 / 8.0), -((fx-1)**2) + (3.0/4.0), (1.0 / 2.0) * ((fx-2) ** 2) - (3.0 / 2.0) * (fx-2) + (9.0 / 8.0)]
    w = tight_quadra_kernal(fx)

    w_grad = tight_quadra_kernal_grad(fx)
    # w_grad = [fx-(3.0/2.0), -2*(fx-1), (fx-2)-(3.0/2.0)]
    vp_grad = ti.Matrix.zero(ti.f32, 2, 2)

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = vec2(i, j)
            if vec2_is_valid(base + offset):
                # weight = w[i][0] * w[j][1]
                weight_grad = vec2(w_grad[i][0]*w[j][1], w[i][0]*w_grad[j][1])
                vp_grad += (u[base + offset] * e).outer_product(weight_grad)
    
    return vp_grad

@ti.func
def gather_vp_grad_v(xp):
    stagger = vec2(0.5, 0.0)
    e = vec2(0.0, 1.0)
    inv_dx = vec2(1.0 / grid_x, 1.0 / grid_y).cast(ti.f32)
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

    # w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Bspline
    # w_grad = [fx-1.5, -2*(fx-1), fx-3.5] # Bspline gradient
    # w = [(1.0 / 2.0) * (fx ** 2) - (3.0 / 2.0) * fx + (9.0 / 8.0), -((fx-1)**2) + (3.0/4.0), (1.0 / 2.0) * ((fx-2) ** 2) - (3.0 / 2.0) * (fx-2) + (9.0 / 8.0)]
    w = tight_quadra_kernal(fx)

    # w_grad = [fx-(3.0/2.0), -2*(fx-1), (fx-2)-(3.0/2.0)]
    w_grad = tight_quadra_kernal_grad(fx)
    vp_grad = ti.Matrix.zero(ti.f32, 2, 2)

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = vec2(i, j)
            if vec2_is_valid(base + offset):
                # weight = w[i][0] * w[j][1]
                weight_grad = vec2(w_grad[i][0]*w[j][1], w[i][0]*w_grad[j][1])
                vp_grad += (v[base + offset] * e).outer_product(weight_grad)
    
    return vp_grad

@ti.kernel
def G2P():
    stagger_u = vec2(0.0, 0.5)
    stagger_v = vec2(0.5, 0.0)
    for p in ti.grouped(particle_positions):
        if particle_type[p] == P_FLUID:
            # update velocity
            xp = particle_positions[p]
            u_pic = gather_vp_u(xp)
            v_pic = gather_vp_v(xp)
            """
            if isnan(u_pic) or isnan(v_pic):
                print("u or v isnan")
                print("u: ", u_pic)
                print("v: ", v_pic)
            """
            new_v_pic = vec2(u_pic, v_pic)
            particle_velocities[p] = new_v_pic

            # update c
            cp_x[p] = gather_cp_x(xp)
            cp_y[p] = gather_cp_y(xp)

            # update T
            particle_last_T[p] = particle_T[p]
            particle_T[p] = gather_Tp(xp)

@ti.kernel
def update_deformation_gradient():
    # stagger_u = vec2(0.0, 0.5)
    # stagger_v = vec2(0.5, 0.0)
    for p in ti.grouped(particle_positions):
        if particle_type[p] == P_FLUID:
            xp = particle_positions[p]
            vp_grad = ti.Matrix.zero(ti.f32, 2, 2)
            vp_grad += gather_vp_grad_u(xp)
            vp_grad += gather_vp_grad_v(xp)
            # update deformation gradient
            count = 1
            while (ti.Matrix.identity(ti.f32, 2) + dt * vp_grad).determinant() <= 0:
                count *= 2
                vp_grad /= 2
            update_term = ti.Matrix.identity(ti.f32, 2)
            for i in range(count):
                update_term = update_term @ (ti.Matrix.identity(ti.f32, 2) + dt * vp_grad)
            
            new_particle_Fe = update_term @ particle_Fe[p]
            if isnan(new_particle_Fe.determinant()):
                print("Gather Fe error")
                print("Update term: ", update_term)
                print("Old particle_Fe: ", particle_Fe[p])
                print("Vp grad: ", vp_grad)
            if particle_Phase[p] == P_FLUID_PHASE:
                new_particle_Fe = ti.math.sqrt(new_particle_Fe.determinant()) * ti.Matrix.identity(ti.f32, 2)
            particle_Fe[p] = new_particle_Fe


@ti.func
def isnan(x):
    return not (x < 0 or 0 < x or x == 0)

@ti.kernel
def advect_particle(dt: ti.f32):
    for p in ti.grouped(particle_positions):
        if particle_type[p] == P_FLUID:
            pos = particle_positions[p]
            pv = particle_velocities[p]
            """
            if isnan(pos[0]) or isnan(pos[1]):
                print("old pos: ", particle_positions[p])
                print("new pos", pos)
                print("vel: ", pv)
            """
            if pos[0] <= grid_x and pv[0] < 0:  # left boundary
                pos[0] = grid_x
                pv[0] = -pv[0]
            if pos[0] >= w - grid_x and pv[0] > 0:  # right boundary
                pos[0] = w - grid_x
                pv[0] = -pv[0]
            if pos[1] <= grid_y and pv[1] < 0:  # bottom boundary
                pos[1] = grid_y
                pv[1] = -pv[1]
            if pos[1] >= h - grid_y and pv[1] > 0:  # top boundary
                pos[1] = h - grid_y
                pv[1] = -pv[1]
            """
            if isnan(pos[0]) or isnan(pos[1]):
                print("old pos(after adjustment): ", particle_positions[p])
                print("new pos(after adjustment): ", pos)
                print("vel(after adjustment): ", pv)
            """
            pos += pv * dt
            particle_positions[p] = pos
            particle_velocities[p] = pv

@ti.kernel
def update_heat_parameters():
    # update temperature, latent, phase...
    # update parameter (mu, lambda, c, k)
    for p in ti.grouped(particle_positions):
        if particle_type[p] == P_FLUID:
            if particle_Phase[p] == P_SOLID_PHASE:
                if particle_T[p] >= freezing_point and particle_U[p] < particle_l[p]:
                    # melting
                    particle_U[p] += particle_c[p] * p_mass * (particle_T[p] - particle_last_T[p])
                    if particle_U[p] > particle_l[p]:
                        particle_Phase[p] = P_FLUID_PHASE
                        particle_mu[p] = mu_fluid_phase_init
                        particle_la[p] = lambda_fluid_phase_init
                        particle_c[p] = c_fluid_phase_init
                        particle_k[p] = k_fluid_phase_init
                        particle_U[p] = particle_l[p]
                        particle_T[p] = freezing_point
                        particle_last_T[p] = freezing_point
                    else:
                        particle_T[p] = particle_last_T[p]
            elif particle_Phase[p] == P_FLUID_PHASE:
                if particle_T[p] <= freezing_point and particle_U[p] > 0:
                    # freezing
                    particle_U[p] += particle_c[p] * p_mass * (particle_T[p] - particle_last_T[p])
                    if  particle_U[p] < 0:
                        particle_Phase[p] = P_SOLID_PHASE
                        particle_mu[p] = mu_solid_phase_init
                        particle_la[p] = lambda_solid_phase_init
                        particle_c[p] = c_solid_phase_init
                        particle_k[p] = k_solid_phase_init
                        particle_U[p] = 0.0
                        particle_T[p] = freezing_point
                        particle_last_T[p] = freezing_point
                    else:
                        particle_T[p] = particle_last_T[p]
#  -------------Main algorithm-----------

def onestep(dt):
    print("\n------------start step-----------------------")
    
    # 1. Update Fe & Fp
    print("-----------start deformation add plasticity---------------")
    deformation_gradient_add_plasticity()
    print("-----------end deformation add plasticity---------------")


    # 2&3. P2G(+ Weight computation)
    print("-----------start clear field---------------")
    clear_field()
    print("-----------end clear field---------------")
    print("-----------start P2G---------------")
    P2G()
    print("-----------end P2G---------------")
    print("-----------start face normalize---------------")
    face_normalize()
    print("-----------end face nomralize---------------")
    print("-----------start cell normalize---------------")
    cell_normalize()
    print("-----------end cell normalize---------------")
    # 4. classify cells
    print("-----------start mark cell---------------")
    mark_cell() # Note: need to revise to the SSCJ14 version
    print("-----------end mark cell---------------")
    print("-----------start assign temperature---------------")
    assign_temperature()
    print("-----------end assign temperature---------------")
    enforce_boundary()
    # Compute grid first due to initialize issue
    # 5. explicitly update velocity (updated by internal & outer force)
    print("-----------start apply force---------------")
    apply_force(dt)
    print("-----------end apply force---------------")
    # 6. grid collision
    print("-----------start enforce boundary 1 ---------------")
    enforce_boundary()
    print("-----------end enforce boundary 1 ---------------")

    # 7. Chorin style projection
    print("------------start collect face vol-----------------")
    collect_face_vol()
    print("------------end collect face vol-----------------")
    print("-----------start solve pressure---------------")
    solve_pressure(dt)
    print("-----------end solve pressure---------------")
    print("-----------start apply pressure---------------")
    apply_pressure(dt)
    print("-----------end apply pressure---------------")
    print("-----------start enforce boundary 2---------------")
    enforce_boundary()
    print("-----------end enforce boundary 2---------------")

    # 8. Solve heat equation
    print("-----------start solve temperature---------------")
    solve_temperature(dt)
    print("-----------end solve temperature---------------")

    print("-----------start enforce boundary 3---------------")
    enforce_boundary()
    print("-----------end enforce boundary 3---------------")
    # 9. G2P
    print("-----------start G2P---------------")
    G2P()
    print("-----------end G2P---------------")
    print("-----------start update deformation gradient---------------")
    update_deformation_gradient()
    print("-----------end update deformation gradient---------------")
    # 10. Update particles' states
    print("-----------start advect particle---------------")
    advect_particle(dt) # move particle & collision handling
    print("-----------end advect particle---------------")
    print("-----------start update haet parameters---------------")
    update_heat_parameters()
    print("-----------end update heat parameters---------------")

    print("------------end step-----------------------\n")


def simulation(max_time, max_step):
    global dt
    # dt = 0.01
    t = 0
    step = 1

    while step < max_step and t < max_time:
        render()
        print("step 1")
        for i in range(substeps):
            onestep(dt)
            # render()
            pv = particle_velocities.to_numpy()
            max_vel = np.max(np.linalg.norm(pv, 2, axis=1))

            print("step = {}, substeps = {}, time = {}, dt = {}, maxv = {}".
                  format(step, i, t, dt, max_vel))

            t += dt

        step += 1

def main():
    init()
    t0 = time.time()
    simulation(40, 240)
    t1 = time.time()
    print("simulation elapsed time = {} seconds".format(t1 - t0))

    video_manager.make_video(gif=True, mp4=True)

if __name__ == '__main__':
    main()