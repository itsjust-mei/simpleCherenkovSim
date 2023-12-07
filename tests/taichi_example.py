import taichi as ti
import numpy as np
import ffpmg


ti.init(arch=ti.gpu)

WIDTH, HEIGHT = 800, 600
fov = 1.0
aspect_ratio = WIDTH / HEIGHT

eye = ti.Vector([0.0, 0.0, 0.0])
look_at = ti.Vector([0.0, 0.0, -1.0])
up = ti.Vector([0.0, 1.0, 0.0])
fov_rad = np.tan(0.5 * fov)

@ti.data_oriented
class Scene:
    def __init__(self):
        self.vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=3)
        self.indices = ti.field(dtype=ti.i32, shape=3)

scene = Scene()

@ti.kernel
def initialize_scene():
    scene.vertices[0] = [0.0, -0.5, -1.0]
    scene.vertices[1] = [-0.5, 0.5, -1.0]
    scene.vertices[2] = [0.5, 0.5, -1.0]
    scene.indices[0] = 0
    scene.indices[1] = 1
    scene.indices[2] = 2

@ti.kernel
def trace_ray(image: ti.types.ndarray(dtype=ti.f32, ndim=2), time: ti.f32):
    for i, j in ti.ndrange(WIDTH, HEIGHT):
        u = (2.0 * i / WIDTH - 1.0) * aspect_ratio * fov_rad
        v = (1.0 - 2.0 * j / HEIGHT) * fov_rad
        direction = ti.Vector([u, v, -1.0]).normalized()
        ray_origin, ray_direction = eye, direction

        vertex0 = scene.vertices[0]
        vertex1 = scene.vertices[1]
        vertex2 = scene.vertices[2]

        e1 = vertex1 - vertex0
        e2 = vertex2 - vertex0
        h = ray_direction.cross(e2)
        a = e1.dot(h)

        f = 1.0 / a
        s = ray_origin - vertex0
        u = f * s.dot(h)

        q = s.cross(e1)
        v = f * ray_direction.dot(q)

        t = f * e2.dot(q)

        hit = (u >= 0.0) and (v >= 0.0) and (u + v <= 1.0) and (t > 0.0)

        image[i, j] = time if hit else 0.0

    # gradients = ti.grad(image)
    # print(gradients)


initialize_scene()

gui = ti.GUI("Differentiable Ray Tracer", (WIDTH, HEIGHT))

for t in range(2):
    image = np.zeros((WIDTH, HEIGHT), dtype=np.float32)
    trace_ray(image, t)

    gradients = ti.grad(scene.vertices.to_numpy())
    print("Gradients:", gradients)

    gui.set_image(image)
    gui.show()

