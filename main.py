import matplotlib.pyplot as plt
import numpy as np

from camera_models import *  # our package


DECIMALS = 2  # how many decimal places to use in print


# %matplotlib notebook
# %load_ext autoreload
# %autoreload 2


## Camera Calibration Matrix ##  <--
FOCAL_LENGTH = 3.0  # focal length
F = 3.0
PX= 2.0  # principal point x-coordinate
PY= 1.0  # principal point y-coordinate
MX = 1.0  # number of pixels per unit distance in image coordinates in x direction
MY = 1.0  # number of pixels per unit distance in image coordinates in y direction
IMAGE_HEIGTH = 4
IMAGE_WIDTH = 6


## each Camera setting ##
## Camera 1
THETA_X1 = np.pi / 2.0  # roll angle
THETA_Y1 = 0.0  # pitch angle
THETA_Z1 = np.pi  # yaw angle
C1 = np.array([3, -5, 2])  # camera centre
## Camera 2
THETA_X2 = np.pi / 2.0  # roll angle
THETA_Y2 = 0.0  # pitch angle
THETA_Z2 = np.pi  # yaw angle
C2 = np.array([3, 5, 2])  # camera centre
## Camera 3
THETA_X3 = np.pi / 2.0  # roll angle
THETA_Y3 = 0.0  # pitch angle
THETA_Z3 = np.pi  # yaw angle
C3 = np.array([-3, -5, 2])  # camera centre
## Camera 4
THETA_X4 = np.pi / 2.0  # roll angle
THETA_Y4 = 0.0  # pitch angle
THETA_Z4 = np.pi  # yaw angle
C4 = np.array([-3, 5, 2])  # camera centre



calibration_kwargs = {"f": FOCAL_LENGTH, "px": PX, "py": PY, "mx": MX, "my": MY}


# Camera 1 : R and P Matrix
rotation_kwargs1 = {"theta_x": THETA_X1, "theta_y": THETA_Y1, "theta_z": THETA_Z1}
projection_kwargs1 = {**calibration_kwargs, **rotation_kwargs1, "C": C1}

# Camera 2 : R and P Matrix
rotation_kwargs2 = {"theta_x": THETA_X2, "theta_y": THETA_Y2, "theta_z": THETA_Z2}
projection_kwargs2 = {**calibration_kwargs, **rotation_kwargs2, "C": C2}

# Camera 3 : R and P Matrix
rotation_kwargs3 = {"theta_x": THETA_X3, "theta_y": THETA_Y3, "theta_z": THETA_Z3}
projection_kwargs3 = {**calibration_kwargs, **rotation_kwargs3, "C": C3}

# Camera 4 : R and P Matrix
rotation_kwargs4 = {"theta_x": THETA_X4, "theta_y": THETA_Y4, "theta_z": THETA_Z4}
projection_kwargs4 = {**calibration_kwargs, **rotation_kwargs4, "C": C4}


## Print Matrix
K = get_calibration_matrix(**calibration_kwargs)
print("Calibration matrix (K):\n", K.round(DECIMALS))


R1 = get_rotation_matrix(**rotation_kwargs1)
print("\nRotation matrix (R_1):\n", R1.round(DECIMALS))
P1 = get_projection_matrix(**projection_kwargs1)
print("\nProjection matrix (P_1):\n", P1.round(DECIMALS))

R2 = get_rotation_matrix(**rotation_kwargs2)
print("\nRotation matrix (R_2):\n", R2.round(DECIMALS))
P2 = get_projection_matrix(**projection_kwargs2)
print("\nProjection matrix (P_2):\n", P2.round(DECIMALS))

R3 = get_rotation_matrix(**rotation_kwargs3)
print("\nRotation matrix (R_3):\n", R3.round(DECIMALS))
P3 = get_projection_matrix(**projection_kwargs3)
print("\nProjection matrix (P_3):\n", P3.round(DECIMALS))

R4 = get_rotation_matrix(**rotation_kwargs4)
print("\nRotation matrix (R_4):\n", R4.round(DECIMALS))
P4 = get_projection_matrix(**projection_kwargs4)
print("\nProjection matrix (P_4):\n", P4.round(DECIMALS))


## World coordinate axis
dx, dy, dz = np.eye(3)
world_frame = ReferenceFrame(
    origin=np.zeros(3),
    dx=dx,
    dy=dy,
    dz=dz,
    name="World",
)

## camera 1 img plane
camera_frame1 = ReferenceFrame(
    origin=C1,
    dx=R1 @ dx,
    dy=R1 @ dy,
    dz=R1 @ dz,
    name="Camera1",
)
Z1 = PrincipalAxis(
    camera_center=C1,
    camera_dz=camera_frame1.dz,
    f=F,
)
image_frame1 = ReferenceFrame(
    origin=Z1.p - camera_frame1.dx * PX - camera_frame1.dy * PY,
    dx=R1 @ dx,
    dy=R1 @ dy,
    dz=R1 @ dz,
    name="Image1",
)
image_plane1 = ImagePlane(
    origin=image_frame1.origin,
    dx=image_frame1.dx,
    dy=image_frame1.dy,
    heigth=IMAGE_HEIGTH,
    width=IMAGE_WIDTH,
    mx=MX,
    my=MY,
)


## camera 2 img plane
camera_frame2 = ReferenceFrame(
    origin=C2,
    dx=R2 @ dx,
    dy=R2 @ dy,
    dz=R2 @ dz,
    name="Camera2",
)
Z2 = PrincipalAxis(
    camera_center=C2,
    camera_dz=camera_frame2.dz,
    f=F,
)
image_frame2 = ReferenceFrame(
    origin=Z2.p - camera_frame2.dx * PX - camera_frame2.dy * PY,
    dx=R2 @ dx,
    dy=R2 @ dy,
    dz=R2 @ dz,
    name="Image2",
)
image_plane2 = ImagePlane(
    origin=image_frame2.origin,
    dx=image_frame2.dx,
    dy=image_frame2.dy,
    heigth=IMAGE_HEIGTH,
    width=IMAGE_WIDTH,
    mx=MX,
    my=MY,
)


## camera 3 img plane
camera_frame3 = ReferenceFrame(
    origin=C3,
    dx=R3 @ dx,
    dy=R3 @ dy,
    dz=R3 @ dz,
    name="Camera3",
)
Z3 = PrincipalAxis(
    camera_center=C3,
    camera_dz=camera_frame3.dz,
    f=F,
)
image_frame3 = ReferenceFrame(
    origin=Z3.p - camera_frame3.dx * PX - camera_frame3.dy * PY,
    dx=R3 @ dx,
    dy=R3 @ dy,
    dz=R3 @ dz,
    name="Image3",
)
image_plane3 = ImagePlane(
    origin=image_frame3.origin,
    dx=image_frame3.dx,
    dy=image_frame3.dy,
    heigth=IMAGE_HEIGTH,
    width=IMAGE_WIDTH,
    mx=MX,
    my=MY,
)


## camera 4 img plane
camera_frame4 = ReferenceFrame(
    origin=C4,
    dx=R4 @ dx,
    dy=R4 @ dy,
    dz=R4 @ dz,
    name="Camera4",
)
Z4 = PrincipalAxis(
    camera_center=C4,
    camera_dz=camera_frame4.dz,
    f=F,
)
image_frame4 = ReferenceFrame(
    origin=Z4.p - camera_frame4.dx * PX - camera_frame4.dy * PY,
    dx=R4 @ dx,
    dy=R4 @ dy,
    dz=R4 @ dz,
    name="Image4",
)
image_plane4 = ImagePlane(
    origin=image_frame4.origin,
    dx=image_frame4.dx,
    dy=image_frame4.dy,
    heigth=IMAGE_HEIGTH,
    width=IMAGE_WIDTH,
    mx=MX,
    my=MY,
)


## world points and squares
image = Image(heigth=IMAGE_HEIGTH, width=IMAGE_WIDTH)
square1 = Polygon(np.array([
    [-1.0, 5.0, 4.0],
    [1.0, 3.0, 5.0],
    [1.0, 2.0, 2.0],
    [-1.0, 4.0, 1.0],
]))
square2 = Polygon(np.array([
    [-2.0, 4.0, 5.0],
    [2.0, 4.0, 5.0],
    [2.0, 4.0, 1.0],
    [-2.0, 4.0, 1.0],
]))


""" ## visualization
fig = plt.figure(figsize=(6, 6))
ax = fig.gca(projection="3d")
world_frame.draw3d()
camera_frame.draw3d()
image_frame.draw3d()
Z.draw3d()
image_plane.draw3d()
square1.draw3d(pi=image_plane.pi, C=C)
square2.draw3d(pi=image_plane.pi, C=C, color="tab:purple")
ax.view_init(elev=45.0, azim=45.0)
ax.set_title("Camera Geometry")
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(IMAGE_WIDTH, IMAGE_HEIGTH))
ax = fig.gca()
image.draw()
square1.draw(**projection_kwargs)
square2.draw(**projection_kwargs, color="tab:purple")
ax.set_title("Projection of Squares in the Image")
plt.tight_layout()
plt.show()
 """