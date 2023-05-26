import taichi as ti

ti.init(arch='cpu')

@ti.pyfunc
def vec2(x, y):
    return ti.Vector([x, y])


@ti.pyfunc
def vec3(x, y, z):
    return ti.Vector([x, y, z])


@ti.kernel
def main():
    xp = vec2(0.7, 0.7)
    inv_dx = vec2(2.0, 2.0)
    stagger = vec2(0.0, 0.5)

    base = (xp * inv_dx - 0.5).cast(ti.i32)
    fx = xp * inv_dx - base.cast(ti.f32)
    print("cell base: ", base)
    print("cell fx: ", fx)
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)
    print("stagger u base: ", base)
    print("stagger u fx: ", fx)

    stagger = vec2(0.5, 0.0)
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)
    print("stagger v base: ", base)
    print("stagger v fx: ", fx)

if __name__ == '__main__':
    main()