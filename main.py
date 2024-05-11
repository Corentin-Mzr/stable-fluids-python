import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cmasher as cmr
import scipy.sparse.linalg as linalg
from scipy import interpolate

# Algorithm parameters
domain_size: float = 1.0
n_points: int = 61
n_iterations: int = 300
time_step: float = 0.1
max_iteration_conjugate_gradient = None

# Fluid parameters
kinematic_viscosity: float = 0.0001

# To save animation
to_save: bool = False
plot_in_loop: bool = True
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'


def sin_forcing(t: float, pt: tuple[float, float]) -> tuple[float, float]:
    decay = np.maximum(2 - 0.5 * t, 0.0)

    forced_val = (
            decay *
            np.where(
                (
                        (np.cos(np.pi * pt[0]) > 0)
                        and
                        (np.cos(np.pi * pt[0]) < 1)
                        and
                        (np.sin(np.pi * pt[1]) > 0)
                        and
                        (np.sin(np.pi * pt[1]) < 1)
                        and
                        (pt[0] - 0.5) ** 2 + (pt[1] - 0.5) ** 2 <= 0.05
                ),
                np.array([np.cos(np.pi * pt[0]) * np.sin(np.pi * pt[1]), 0.0]),
                np.array([0.0, 0.0]),
            )
    )

    return forced_val


def random_forcing(t: float, pt: tuple[float, float]) -> tuple[float, float]:
    decay = np.maximum(2 - 0.5 * t, 0.0)

    forced_val = (
            decay *
            np.where(
                (
                        (pt[0] > 0.45 * domain_size)
                        and
                        (pt[0] < 0.55 * domain_size)
                        and
                        (pt[1] > 0.45 * domain_size)
                        and
                        (pt[1] < 0.55 * domain_size)
                ),
                np.random.normal(loc=-2.0, scale=0.5, size=2),
                np.array([0.0, 0.0]),
            )
    )

    return forced_val


def forcing(t: float, pt: tuple[float, float]) -> tuple[float, float]:
    decay = np.maximum(2 - 0.5 * t, 0.0)

    forced_val = (
            decay *
            np.where(
                (
                        (pt[0] > 0.4 * domain_size)
                        and
                        (pt[0] < 0.6 * domain_size)
                        and
                        (pt[1] > 0.1 * domain_size)
                        and
                        (pt[1] < 0.3 * domain_size)
                ),
                np.array([0.0, 1.0]),
                np.array([0.0, 0.0]),
            )

            +
            decay *
            np.where(
                (
                        (pt[0] > 0.4 * domain_size)
                        and
                        (pt[0] < 0.6 * domain_size)
                        and
                        (pt[1] > 0.7 * domain_size)
                        and
                        (pt[1] < 0.9 * domain_size)
                ),
                np.array([0.0, -1.0]),
                np.array([0.0, 0.0]),
            )
    )
    return forced_val


def main():
    matplotlib.use('TkAgg')
    plt.style.use("dark_background")

    curve_data: list[np.ndarray] = []
    vel_data: list[np.ndarray] = []

    elt_length: float = domain_size / (n_points - 1)

    # Shape of data
    scalar_shape: tuple[int, int] = (n_points, n_points)
    scalar_dof: int = n_points ** 2

    vector_shape: tuple[int, int, int] = (n_points, n_points, 2)
    vector_dof: int = 2 * n_points ** 2

    x: np.ndarray = np.linspace(start=0.0, stop=domain_size, num=n_points)
    y: np.ndarray = np.linspace(start=0.0, stop=domain_size, num=n_points)

    # "ij" for diff operators
    xx, yy = np.meshgrid(x, y, indexing="ij")

    coords: np.ndarray = np.concatenate(
        (xx[..., np.newaxis],
         yy[..., np.newaxis]),
        axis=-1)

    forcing_vectorized = np.vectorize(pyfunc=sin_forcing, signature='(),(d)->(d)')

    def partial_derivative_x(field: np.ndarray) -> np.ndarray:
        diff = np.zeros_like(field)
        diff[1:-1, 1:-1] = (field[2:, 1:-1] - field[0:-2, 1:-1]) / (2 * elt_length)

        return diff

    def partial_derivative_y(field: np.ndarray) -> np.ndarray:
        diff = np.zeros_like(field)
        diff[1:-1, 1:-1] = (field[1:-1, 2:] - field[1:-1, 0:-2]) / (2 * elt_length)

        return diff

    def laplace(field: np.ndarray) -> np.ndarray:
        """
        :param field: The field we want to compute the laplacian of
        :return: Computed laplacian of the field
        """
        diff = np.zeros_like(field)

        diff[1:-1, 1:-1] = (field[0:-2, 1:-1] +
                            field[1:-1, 0:-2] -
                            4 * field[1:-1, 1:-1] +
                            field[2:, 1:-1] +
                            field[1:-1, 2:]) / (elt_length ** 2)

        return diff

    def divergence(vector_field: np.ndarray) -> np.ndarray:
        """
        :param vector_field:
        :return: Divergence operator
        """
        div = partial_derivative_x(vector_field[..., 0]) + partial_derivative_y(vector_field[..., 1])
        return div

    def gradient(field: np.ndarray) -> np.ndarray:
        """
        :param field:
        :return: Gradient operator
        """
        gradient_applied = np.concatenate((partial_derivative_x(field)[..., np.newaxis],
                                           partial_derivative_y(field)[..., np.newaxis]),
                                          axis=-1)
        return gradient_applied

    def curl(vector_field: np.ndarray) -> np.ndarray:
        curl_applied = partial_derivative_x(vector_field[..., 1]) - partial_derivative_y(vector_field[..., 0])
        return curl_applied

    def advect(field: np.ndarray, vector_field: np.ndarray) -> np.ndarray:
        """
        :param field: A field
        :param vector_field: A vector field
        :return: Computed advection of the field
        """
        backtrack_pos: np.ndarray = np.clip((coords - time_step * vector_field), a_min=0.0, a_max=domain_size)
        advected_field: np.ndarray = interpolate.interpn(points=(x, y), values=field, xi=backtrack_pos)

        return advected_field

    def diffusion(vector_field_flat: np.ndarray) -> np.ndarray:
        """
        :param vector_field_flat:
        :return: Diffusion operator
        """
        vector_field: np.ndarray = vector_field_flat.reshape(vector_shape)
        diffusion_applied: np.ndarray = vector_field - kinematic_viscosity * time_step * laplace(vector_field)

        return diffusion_applied.flatten()

    def poisson(field_flat: np.ndarray) -> np.ndarray:
        field = field_flat.reshape(scalar_shape)
        poisson_applied = laplace(field)
        return poisson_applied.flatten()

    vel_prev: np.ndarray = np.zeros(vector_shape)

    current_time: float = 0.0
    # Main loop
    print("Doing main loop...")
    for i in range(n_iterations):
        current_time += time_step

        forces: np.ndarray = forcing_vectorized(current_time, coords)

        # Apply forces
        vel_with_force: np.ndarray = vel_prev + time_step * forces

        # Convection (self advection)
        vel_advected: np.ndarray = advect(field=vel_with_force, vector_field=vel_with_force)

        # Diffusion
        vel_diffused = linalg.cg(
            A=linalg.LinearOperator(
                shape=(vector_dof, vector_dof),
                matvec=diffusion,
            ),
            b=vel_advected.flatten(),
            maxiter=max_iteration_conjugate_gradient

        )[0].reshape(vector_shape)

        # Pressure correction
        pressure = linalg.cg(
            A=linalg.LinearOperator(
                shape=(scalar_dof, scalar_dof),
                matvec=poisson,
            ),
            b=divergence(vel_diffused).flatten(),
            maxiter=max_iteration_conjugate_gradient
        )[0].reshape(scalar_shape)

        # Velocity correction for incompressibility
        vel_projected = vel_diffused - gradient(pressure)

        # Update for next iteration
        vel_prev = vel_projected

        # Fluid movement
        curve = curl(vel_projected)

        # Save data
        curve_data.append(curve)
        vel_data.append(vel_projected)

        # Plot
        if plot_in_loop:
            plt.contourf(xx, yy, curve, cmap=cmr.redshift, levels=100, vmin=-np.max(curve), vmax=np.max(curve))
            # plt.quiver(xx, yy, vel_projected[..., 0], vel_projected[..., 1], color="dimgray")
            plt.draw()
            plt.pause(0.0001)
            plt.clf()
    print("End of main loop !")

    def update(i: int):
        """ Update graph """
        plt.cla()
        im = ax.contourf(xx, yy, curve_data[i], cmap=cmr.redshift, levels=100)
        return im

    if to_save:
        fig, ax = plt.subplots(figsize=(3, 3), dpi=160)
        ani = FuncAnimation(fig=fig,
                            func=update,
                            frames=len(curve_data),
                            interval=time_step,
                            blit=False)

        save_file = r"C:\Users\deads\Bureau\fluid_simulation.mp4"
        ani.save(save_file, writer=matplotlib.animation.FFMpegWriter(fps=30))
        print("Animation saved !")


if __name__ == "__main__":
    print("Starting program")
    main()
    print("End of program")
