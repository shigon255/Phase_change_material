import taichi as ti
# from CGSolver import CGSolver
from Pressure_CGSolver import Pressure_CGSolver
from Temperature_CGSolver import Temperature_CGSolver
from MICPCGSolver import MICPCGSolver
from MGPCGSolver import MGPCGSolver
import numpy as np
from utils import ColorMap, vec2, vec3, clamp
import utils
import time

ti.init(arch=ti.gpu, default_fp=ti.f32, debug = True)

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

p_vol, p_rho = (grid_x * 0.5)**2, 1
p_mass = p_vol * p_rho
g = -9.8
substeps = 4

# Young's modulas , Poisson's ratio and lame parameter
E_ice, nu_ice = 9, 0.3  # Young's modulus(Gpa) and Poisson's ratio for ice 
mu_ice, lambda_ice = E_ice / (2 * (1 + nu_ice)), E_ice * nu_ice / (
    (1 + nu_ice) * (1 - 2 * nu_ice))  # Lame parameters

E_fluid, nu_fluid = 2, 0.45  # Young's modulus(Gpa) and Poisson's ratio for ice 
mu_fluid, lambda_fluid = E_fluid / (2 * (1 + nu_fluid)), E_fluid * nu_fluid / (
    (1 + nu_fluid) * (1 - 2 * nu_fluid))  # Lame parameters

# Initial value
T_air_init = 343 # 343K
T_solid_init = 373 # 373K
T_fluid_init = 273 # freezing point

# heat capacity
c_ice = 2.093 # J/g*c
c_fluid = 4.182 # J/g*c

# heat conductivity
k_ice = 2
k_fluid = 0.606

# phase change parameters
freezing_point = 273
ice_latent = 334

# ----------------fields---------------
# face part
# velocity field
u = ti.field(dtype=ti.f32, shape=(m + 1, n))
v = ti.field(dtype=ti.f32, shape=(m, n + 1))
u_face_mass = ti.field(dtype=ti.f32, shape=(m + 1, n))
v_face_mass = ti.field(dtype=ti.f32, shape=(m, n + 1))
u_temp = ti.field(dtype=ti.f32, shape=(m + 1, n))
v_temp = ti.field(dtype=ti.f32, shape=(m, n + 1))

# internal force
fu = ti.field(dtype=ti.f32, shape=(m + 1, n))
fv = ti.field(dtype=ti.f32, shape=(m, n + 1))

# heat conductivity field
k_u = ti.field(dtype=ti.f32, shape=(m + 1, n))
k_v = ti.field(dtype=ti.f32, shape=(m, n + 1))

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
particle_Fp = ti.Matrix.field(2, 2, dtype=ti.f32, shape=(m, n, npar, npar))  # Plastic deformation gradient (F_E)

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
P_FLUID_PHASE = 2
P_SOLID_PHASE = 3

# -----------------params in render-----------
screen_res = (800, 800 * n // m)
bwrR = ColorMap(1.0, .25, 1, .5)
bwrG = ColorMap(1.0, .5, .5, .5)
bwrB = ColorMap(1.0, 1, .25, .5)
color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=screen_res)
gui = ti.GUI("watersim2D", screen_res, show_gui = False)

# cell type
cell_type = ti.field(dtype=ti.i32, shape=(m, n))


#pressure solver
# preconditioning = 'MG'
preconditioning = None

MIC_blending = 0.97

mg_level = 4
pre_and_post_smoothing = 2
bottom_smoothing = 10

dt = 0.01

if preconditioning == None:
    # solver = CGSolver(m, n, u, v, cell_type)
    pressure_solver = Pressure_CGSolver(m, n, u, v, dt, Jp, Je, inv_lambda, cell_type)
    temperature_solver = Temperature_CGSolver(m, n, k_u, k_v, T, c, cell_type)
elif preconditioning == 'MIC':
    solver = MICPCGSolver(m, n, u, v, cell_type, MIC_blending=MIC_blending)
elif preconditioning == 'MG':
    solver = MGPCGSolver(m,
                         n,
                         u,
                         v,
                         cell_type,
                         multigrid_level=mg_level,
                         pre_and_post_smoothing=pre_and_post_smoothing,
                         bottom_smoothing=bottom_smoothing)

# extrap utils
valid = ti.field(dtype=ti.i32, shape=(m + 1, n + 1))
valid_temp = ti.field(dtype=ti.i32, shape=(m + 1, n + 1))

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

            color_buffer[i, j] = map_color(m * 0.5)

    def render_pixels():
        fill_marker()
        img = color_buffer.to_numpy()
        gui.set_image(img) 

    def render_particles():
        bg_color = 0x112f41
        particle_color = 0x068587
        particle_radius = 1.0

        pf = particle_type.to_numpy()
        np_type = pf.copy()
        np_type = np.reshape(np_type, -1)

        pos = particle_positions.to_numpy()
        np_pos = pos.copy()
        np_pos = np.reshape(pos, (-1, 2))
        np_pos = np_pos[np.where(np_type == P_FLUID)]

        for i in range(np_pos.shape[0]):
            np_pos[i][0] /= w
            np_pos[i][1] /= h

        gui.clear(bg_color)
        gui.circles(np_pos, radius=particle_radius, color=particle_color)
    
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

        for i, j in p:
            p[i, j] = 0.0
        
        for i, j in k_u:
            k_u[i, j] = c_ice # for ice, 2.18 W/mK at 273K

        for i, j in k_v:
            k_v[i, j] = c_ice # for ice, 2.18 W/mK at 273K
        
        for i, j in fu:
            fu[i, j] = 0.0
        
        for i, j in fv:
            fv[i, j] = 0.0

        for i, j in J:
            J[i, j] = 1
            Je[i, j] = 1
            c[i, j] = c_ice # for ice, 2093 J/K
            T[i, j] = T_fluid_init # at 273 K
            inv_lambda[i, j] = 1 / lambda_ice

    @ti.kernel
    def init_particles():
        for i, j, ix, jx in particle_positions:
            if cell_type[i, j] == utils.FLUID:
                particle_type[i, j, ix, jx] = P_FLUID
            else:
                particle_type[i, j, ix, jx] = 0

            px = i * grid_x + (ix + ti.random(ti.f32)) * pspace_x
            py = j * grid_y + (jx + ti.random(ti.f32)) * pspace_y

            particle_positions[i, j, ix, jx] = vec2(px, py)
            particle_velocities[i, j, ix, jx] = vec2(0.0, 0.0)
            cp_x[i, j, ix, jx] = vec2(0.0, 0.0)
            cp_y[i, j, ix, jx] = vec2(0.0, 0.0)
            particle_Fe[i, j, ix, jx] = ti.Matrix.identity(dt = ti.f32, n=2)
            particle_Fp[i, j, ix, jx] = ti.Matrix.identity(dt = ti.f32, n=2)
            particle_mu[i, j, ix, jx] = mu_ice
            particle_la[i, j, ix, jx] = lambda_ice
            particle_T[i, j, ix, jx] = T_fluid_init
            particle_last_T[i, j, ix, jx] = T_fluid_init
            particle_U[i, j, ix, jx] = 0  # [0, Lp], initialilly at 0
            particle_c[i, j, ix, jx] = c_ice # for ice J/g * C
            particle_k[i, j, ix, jx] = k_ice
            particle_l[i, j, ix, jx] = p_mass * ice_latent # for ice 334J/g to melt
            particle_Phase[i, j, ix, jx] = P_SOLID_PHASE


    # init_dambreak(4, 4)
    init_spherefall(5,3,2)
    init_field()
    init_particles()

# -------------subprocesses----------------
@ti.kernel
def deformation_gradient_add_plasticity():
    for p in ti.grouped(particle_positions):
        U, sig, V = ti.svd(particle_Fe[p])
        for d in ti.static(range(2)):
            sig[d, d] = ti.min(ti.max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
        Jp = particle_Fp[p].determinant()
        # Note: this part is (Jp)^(1/d), here d = 2
        particle_Fe[p] = ti.sqrt(Jp) * particle_Fe[p]
        particle_Fp[p] = (1 / ti.sqrt(Jp)) * particle_Fp[p]

@ti.func
def scatter_face(grid_v, grid_m, grid_k, grid_f, xp, vp, cp, fp, stagger, kp, e):
    inv_dx = vec2(inv_grid_x, inv_grid_y).cast(ti.f32)
    # inv_dx = vec2(1.0 / grid_x, 1.0 / grid_y).cast(ti.f32)
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

    # Note, in SSJCTS14, they use cubic B-spline
    w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Quadratic Bspline
    w_grad = [fx-1.5, -2*(fx-1), fx-3.5] # Bspline gradient
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = vec2(i, j)
            dpos = (offset.cast(ti.f32) - fx) * vec2(grid_x, grid_y)
            weight_grad = vec2(w_grad[i][0]*w[j][1], w[i][0]*w_grad[j][1])
            weight = w[i][0] * w[j][1] # x, y directions, respectively
            grid_v[base + offset] += weight * p_mass * (vp + cp.dot(dpos))
            grid_k[base + offset] += weight * p_mass * kp   # Need not to multiply affine to heat conductivity(maybe?)
            grid_m[base + offset] += weight * p_mass # Maybe in waterSim2d, they assume that every particles' mass is 1?
            grid_f[base + offset] += ti.math.dot(e, (-fp) @ weight_grad )

@ti.func
def scatter_cell(cell_m,  cell_J, cell_Je, cell_Jp, cell_c, cell_T, cell_inv_lambda, xp, par_J, par_Je, par_Jp, par_c, par_T, par_inv_lambda):
    inv_dx = vec2(inv_grid_x, inv_grid_y).cast(ti.f32)
    base = (xp * inv_dx - 0.5).cast(ti.i32)
    fx = xp * inv_dx - base.cast(ti.f32)
    # Note, in SSJCTS14, they use cubic B-splines
    w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2] # Quadratic Bspline
    # inv_dx = vec2(1.0 / grid_x, 1.0 / grid_y).cast(ti.f32)
    # base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    # fx = xp * inv_dx - (base.cast(ti.f32) + stagger)
    # w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Quadratic Bspline

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = vec2(i, j)
            # dpos = (offset.cast(ti.f32) - fx) * vec2(grid_x, grid_y)
            weight = w[i][0] * w[j][1] # x, y directions, respectively
            cell_m[base + offset] += weight * p_mass
            cell_Je[base + offset] += weight * p_mass * par_Je
            cell_J[base + offset] += weight * p_mass * par_J
            cell_Jp[base + offset] += weight * p_mass * (par_Jp)
            cell_c[base + offset] += weight * p_mass * par_c
            cell_T[base + offset] += weight * p_mass * par_T
            cell_inv_lambda[base + offset] += weight * p_mass * par_inv_lambda
            



@ti.kernel
def P2G():
    stagger_u = vec2(0.0, 0.5)
    stagger_v = vec2(0.5, 0.0)
    for p in ti.grouped(particle_positions):
        if particle_type[p] == P_FLUID:
            xp = particle_positions[p]
            par_F = particle_Fe[p] * particle_Fp[p]
            par_Je = particle_Fe[p].determinant()
            par_Jp = particle_Fp[p].determinant()
            par_J = par_Je * par_Jp
            U, sig, V = ti.svd(par_F)
            # P(F) F^T
            par_f = 2 * particle_mu[p] * (par_F - U@V.transpose()) + ti.Matrix.identity(float, 2) * particle_la[p] * par_J * (par_J-1)
            par_f = p_vol * par_f # f is 2 by 2 matrix, whilch will be multiplid by weight gradient
            # take the first row and second row to compute the force in u direction and v direction respectively
            # face
            scatter_face(u, u_face_mass, k_u, fu, xp, particle_velocities[p][0],
                            cp_x[p], par_f.transpose(), stagger_u, particle_k[p], vec2(1.0, 0.0))
            scatter_face(v, v_face_mass, k_v, fv, xp, particle_velocities[p][1],
                            cp_y[p], par_f.transpose(), stagger_v, particle_k[p], vec2(0.0, 1.0))
            # cell
            # par_Je = particle_Fe[p].determinant()
            # par_Jp = particle_Fp[p].determinant()
            # par_J = par_Je * par_Jp
            par_c = particle_c[p]
            par_T = particle_T[p]
            par_inv_lambda = 1.0 / particle_la[p]
            scatter_cell(cell_mass, J, Je, Jp, c, T, inv_lambda, xp, par_J, par_Je, par_Jp, par_c, par_T, par_inv_lambda)

@ti.kernel
def clear_field():
    u.fill(0.0)
    v.fill(0.0)
    u_face_mass.fill(0.0)
    v_face_mass.fill(0.0)
    k_u.fill(0.0)
    k_v.fill(0.0)
    cell_mass.fill(0.0)

@ti.func
def is_valid(i, j):
    return i >= 0 and i < m and j >= 0 and j < n


@ti.func
def is_fluid(i, j):
    return is_valid(i, j) and cell_type[i, j] == utils.FLUID


@ti.func
def is_solid(i, j):
    return is_valid(i, j) and cell_type[i, j] == utils.SOLID


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
            mass_thres = 1e-6
            if u_face_mass[i, j] > mass_thres and u_face_mass[i+1, j] > mass_thres and v_face_mass[i, j] > mass_thres and v_face_mass[i, j+1] > mass_thres:
                cell_type[i, j] = utils.FLUID
            else:
                cell_type[i, j] = utils.AIR
@ti.kernel
def assign_temperature():
    for i, j in cell_type:
        if cell_type == utils.AIR:
            # air: 60 C
            T[i, j] = 343 
        else:
            # solid(edge): 100 C
            T[i, j] = 373


@ti.kernel
def face_normalize():
    for i, j in u:
        if u_face_mass[i, j] > 0:
            u[i, j] = u[i, j] / u_face_mass[i, j]

    for i, j in v:
        if v_face_mass[i, j] > 0:
            v[i, j] = v[i, j] / v_face_mass[i, j]

    for i, j in k_u:
        if u_face_mass[i, j] > 0:
            k_u[i, j] = k_u[i, j] / u_face_mass[i, j]

    for i, j in k_v:
        if v_face_mass[i, j] > 0:
            k_v[i, j] = k_v[i, j] / v_face_mass[i, j]

@ti.kernel
def cell_normalize():
    for i, j in J:
        if cell_mass[i, j] > 0:
            J[i, j] = J[i, j] / cell_mass[i, j]

    for i, j in Je:
        if cell_mass[i, j] > 0:
            Je[i, j] = Je[i, j] / cell_mass[i, j]

    for i, j in Jp:
        if cell_mass[i, j] > 0:
            Jp[i, j] = J[i, j] / cell_mass[i, j]

    for i, j in c:
        if cell_mass[i, j] > 0:
            c[i, j] = c[i, j] / cell_mass[i, j]

    for i, j in T:
        if cell_mass[i, j] > 0:
            T[i, j] = T[i, j] / cell_mass[i, j]

    for i, j in inv_lambda:
        if cell_mass[i, j] > 0:
            inv_lambda[i, j] = inv_lambda[i, j] / cell_mass[i, j]
    
@ti.kernel
def apply_force(dt: ti.f32):

    # internal force (have been calculated in P2G)
    for i, j in u:
        u[i, j] += fu[i, j] * dt
    
    for i, j in v:
        v[i, j] += fv[i, j] * dt

    # gravity(only v direction)
    for i, j in v:
        v[i, j] += g * dt
        
@ti.kernel
def enforce_boundary():
    # u solid
    for i, j in u:
        if is_solid(i - 1, j) or is_solid(i, j):
            u[i, j] = 0.0

    # v solid
    for i, j in v:
        if is_solid(i, j - 1) or is_solid(i, j):
            v[i, j] = 0.0

def solve_pressure(dt):
    scale_A = dt / (p_rho * grid_x * grid_x)
    scale_b = 1 / grid_x

    pressure_solver.system_init(scale_A, scale_b)
    pressure_solver.solve(500)

    p.copy_from(pressure_solver.p)

@ti.kernel
def apply_pressure(dt: ti.f32):
    scale = dt / (p_rho * grid_x)

    for i, j in ti.ndrange(m, n):
        if is_fluid(i - 1, j) or is_fluid(i, j):
            if is_solid(i - 1, j) or is_solid(i, j):
                u[i, j] = 0
            else:
                u[i, j] -= scale * (p[i, j] - p[i - 1, j])

        if is_fluid(i, j - 1) or is_fluid(i, j):
            if is_solid(i, j - 1) or is_solid(i, j):
                v[i, j] = 0
            else:
                v[i, j] -= scale * (p[i, j] - p[i, j - 1])

def solve_temperature(dt):
    scale_A = dt / (p_rho * grid_x * grid_x)
    # scale_b = 1 / grid_x
    scale_b = 1

    temperature_solver.system_init(scale_A, scale_b)
    temperature_solver.solve(500)

    T.copy_from(temperature_solver.p)

@ti.func
def gather_vp(grid_v, xp, stagger):
    inv_dx = vec2(1.0 / grid_x, 1.0 / grid_y).cast(ti.f32)
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

    w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Bspline

    v_pic = 0.0

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = vec2(i, j)
            weight = w[i][0] * w[j][1]
            v_pic += weight * grid_v[base + offset]

    return v_pic


@ti.func
def gather_cp(grid_v, xp, stagger):
    inv_dx = vec2(1.0 / grid_x, 1.0 / grid_y).cast(ti.f32)
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

    w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Bspline
    w_grad = [fx-1.5, -2*(fx-1), fx-3.5] # Bspline gradient

    cp = vec2(0.0, 0.0)

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = vec2(i, j)
            # dpos = offset.cast(ti.f32) - fx
            # weight = w[i][0] * w[j][1]
            # cp += 4 * weight * dpos * grid_v[base + offset] * inv_dx[0]
            weight_grad = vec2(w_grad[i][0]*w[j][1], w[i][0]*w_grad[j][1])
            cp += weight_grad * grid_v[base + offset]

    return cp

@ti.func
def gather_Tp(grid_T, xp):
    inv_dx = vec2(inv_grid_x, inv_grid_y).cast(ti.f32)
    base = (xp * inv_dx - 0.5).cast(ti.i32)
    fx = xp * inv_dx - base.cast(ti.f32)

    w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Bspline

    Tp = 0.0

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = vec2(i, j)
            # dpos = offset.cast(ti.f32) - fx
            weight = w[i][0] * w[j][1]
            # cp += 4 * weight * dpos * grid_v[base + offset] * inv_dx[0]
            Tp += weight * grid_T[base + offset]

    return Tp

@ti.func
def gather_vp_grad(grid_v, xp, stagger, e):
    inv_dx = vec2(1.0 / grid_x, 1.0 / grid_y).cast(ti.f32)
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

    w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Bspline
    w_grad = [fx-1.5, -2*(fx-1), fx-3.5] # Bspline gradient
    vp_grad = ti.Matrix.zero(float, 2, 2)

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = vec2(i, j)
            # weight = w[i][0] * w[j][1]
            weight_grad = vec2(w_grad[i][0]*w[j][1], w[i][0]*w_grad[j][1])
            vp_grad += (grid_v[base + offset] * e).outer_product(weight_grad)
    
    return vp_grad

@ti.kernel
def G2P():
    stagger_u = vec2(0.0, 0.5)
    stagger_v = vec2(0.5, 0.0)
    for p in ti.grouped(particle_positions):
        if particle_type[p] == P_FLUID:
            # update velocity
            xp = particle_positions[p]
            u_pic = gather_vp(u, xp, stagger_u)
            v_pic = gather_vp(v, xp, stagger_v)
            new_v_pic = vec2(u_pic, v_pic)
            particle_velocities[p] = new_v_pic

            # update c
            cp_x[p] = gather_cp(u, xp, stagger_u)
            cp_y[p] = gather_cp(v, xp, stagger_v)

            # update T
            particle_last_T[p] = particle_T[p]
            particle_T[p] = gather_Tp(T, xp)

@ti.kernel
def update_deformation_gradient():
    stagger_u = vec2(0.0, 0.5)
    stagger_v = vec2(0.5, 0.0)
    for p in ti.grouped(particle_positions):
        if particle_type[p] == P_FLUID:
            xp = particle_positions[p]
            vp_grad = ti.Matrix.zero(float, 2, 2)
            vp_grad += gather_vp_grad(u, xp, stagger_u, vec2(1.0, 0.0))
            vp_grad += gather_vp_grad(v, xp, stagger_v, vec2(0.0, 1.0))
            # update deformation gradient
            new_particle_Fe = (ti.Matrix.identity(float, 2) + vp_grad) * particle_Fe[p]
            if particle_Phase[p] == P_FLUID_PHASE:
                new_particle_Fe = ti.math.pow(new_particle_Fe.determinant(), 0.5) * ti.Matrix.identity(dt = ti.f32, n=2)
            particle_Fe[p] = new_particle_Fe

@ti.kernel
def advect_particle(dt: ti.f32):
    for p in ti.grouped(particle_positions):
        if particle_type[p] == P_FLUID:
            pos = particle_positions[p]
            pv = particle_velocities[p]

            pos += pv * dt

            if pos[0] <= grid_x:  # left boundary
                pos[0] = grid_x
                pv[0] = 0
            if pos[0] >= w - grid_x:  # right boundary
                pos[0] = w - grid_x
                pv[0] = 0
            if pos[1] <= grid_y:  # bottom boundary
                pos[1] = grid_y
                pv[1] = 0
            if pos[1] >= h - grid_y:  # top boundary
                pos[1] = h - grid_y
                pv[1] = 0

            particle_positions[p] = pos
            particle_velocities[p] = pv

@ti.kernel
def update_heat_parameters():
    # update temperature, latent, phase...
    # update parameter (mu, lambda, c, k)
    for p in ti.grouped(particle_positions):
        if particle_Phase[p] == P_SOLID_PHASE:
            if particle_T[p] >= freezing_point and particle_U[p] < particle_l[p]:
                # melting
                particle_U[p] += particle_c[p] * p_mass * (particle_T[p] - particle_last_T[p])
                if particle_U[p] > particle_l[p]:
                    particle_Phase[p] = P_FLUID_PHASE
                    particle_mu[p] = mu_fluid
                    particle_la[p] = lambda_fluid
                    particle_c[p] = c_fluid
                    particle_k[p] = k_fluid
                else:
                    particle_T[p] = particle_last_T[p]
        elif particle_Phase[p] == P_FLUID_PHASE:
            if particle_T[p] <= freezing_point and particle_U[p] > 0:
                # freezing
                particle_U[p] += particle_c[p] * p_mass * (particle_T[p] - particle_last_T[p])
                if  particle_U[p] < 0:
                    particle_Phase[p] = P_FLUID_PHASE
                    particle_mu[p] = mu_fluid
                    particle_la[p] = lambda_fluid
                    particle_c[p] = c_fluid
                    particle_k[p] = k_fluid
                else:
                    particle_T[p] = particle_last_T[p]
            
#  -------------Main algorithm-----------

def onestep(dt):
    # Compute grid first due to initialize issue
    # 5. explicitly update velocity (updated by internal & outer force)
    apply_force(dt)
    # 6. grid collision
    enforce_boundary()

    # 7. Chorin style projection

    solve_pressure(dt)
    apply_pressure(dt)
    enforce_boundary()

    # 8. Solve heat equation
    solve_temperature(dt)

    # 9. G2P
    G2P()
    update_deformation_gradient()
    # 10. Update particles' states
    advect_particle(dt) # move particle & collision handling
    update_heat_parameters()

    # 1. Update Fe & Fp
    deformation_gradient_add_plasticity()


    # 2&3. P2G(+ Weight computation)
    clear_field()
    P2G()
    face_normalize()
    cell_normalize()
    # 4. classify cells
    mark_cell() # Note: need to revise to the SSCJ14 version
    assign_temperature()
    # enforce_boundary()


def simulation(max_time, max_step):
    global dt
    # dt = 0.01
    t = 0
    step = 1

    while step < max_step and t < max_time:
        render()

        for i in range(substeps):
            onestep(dt)

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