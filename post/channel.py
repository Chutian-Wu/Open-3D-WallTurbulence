import numpy as np
import h5py
import toml
from scipy.fft import fftshift, fft, ifft, fftfreq
import compact_difference as cd


def s2p(u_fft):
    [NZ, NXH, NY] = np.shape(u_fft)
    NX = NXH * 2
    u = np.zeros((NZ, NX, NY), dtype=np.complex128)
    u[:, :NXH, :] = ifft(u_fft, axis=0, norm='forward')
    u[:, NXH + 1:, :] = np.conjugate(np.flip(u[:, 1:NXH, :], axis=1))
    u = ifft(u, axis=1, norm='forward')
    return np.real(u)


class CHANNEL:

    def __init__(self, dkz, dkx, alpha, Nz, Nx, Ny, nu):
        self.dkx = dkx
        self.dkz = dkz
        self.Lx = 2 * np.pi / dkx
        self.Lz = 2 * np.pi / dkz
        self.nu = nu
        self.Re = 1.0 / nu
        self.Nx = Nx
        self.Nz = Nz
        self.Ny = Ny
        self.x = np.arange(Nx) * self.Lx / Nx
        self.z = np.arange(Nz) * self.Lz / Nz
        self.y = np.tanh(alpha * np.linspace(-1, 1, Ny)) / np.tanh(alpha) + 1
        self.NxH = int(Nx // 2)
        self.NzH = int(Nz // 2)
        self.NyH = int(Ny // 2)
        self.kx = np.linspace(0, self.NxH - 1, self.NxH) * dkx
        self.kz = fftshift(fftfreq(Nz, d=1.0 / (Nz * dkz)))
        self.var_dict = {}

        # derivative over y
        D1 = cd.operator_D1(Ny, 2.0 / (Ny - 1.0))
        D2 = cd.operator_D2(Ny, 2.0 / (Ny - 1.0))
        D3 = cd.operator_D3(Ny, 2.0 / (Ny - 1.0))
        D4 = cd.operator_D4(Ny, 2.0 / (Ny - 1.0))

        self.D1 = D1
        self.D2 = D2
        self.D3 = D3
        self.D4 = D4

        md = cd.mesh_tanh(Ny, alpha=alpha)
        self.DY1 = D1 * md['c11'][:, np.newaxis]

        self.DY2 = D2 * md['c22'][:, np.newaxis]
        self.DY2 += D1 * md['c21'][:, np.newaxis]

        self.DY3 = D3 * md['c33'][:, np.newaxis]
        self.DY3 += D2 * md['c32'][:, np.newaxis]
        self.DY2 += D1 * md['c31'][:, np.newaxis]

        self.DY4 = D4 * md['c44'][:, np.newaxis]
        self.DY4 += D3 * md['c43'][:, np.newaxis]
        self.DY4 += D2 * md['c42'][:, np.newaxis]
        self.DY4 += D1 * md['c41'][:, np.newaxis]

    def add_var(self, name_, value_):
        self.var_dict[name_] = value_

    def get_var(self, name_):
        return self.var_dict.get(name_, f'key {name_} not found')

    def add_var_frdir(self, path):

        def load_complex(data, var):
            """Helper function to load complex data from HDF5."""
            if f'{var}_real' in data and f'{var}_imag' in data:
                return np.array(
                    data[f'{var}_real']) + 1j * np.array(data[f'{var}_imag'])
            return None

        var_names = [
            'velx', 'vely', 'velz', 'vorx', 'vory', 'vorz', 'lmbx', 'lmby',
            'lmbz'
        ]

        with h5py.File(path, 'r') as data:
            for var in var_names:
                value = load_complex(data, var)
                self.add_var(f'{var}_fft', value)

    def write_vtk(self, vtk_filename, var_name_list):
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        # Open the file in binary mode
        with open(vtk_filename, 'wb') as f:
            # Write VTK header
            f.write(b"# vtk DataFile Version 3.0\n")
            f.write(b"3D vector field (structured grid)\n")
            f.write(b"BINARY\n")
            f.write(b"DATASET STRUCTURED_GRID\n")

            # Specify grid dimensions
            f.write(f"DIMENSIONS {Nx} {Ny} {Nz}\n".encode())

            # Write grid points explicitly
            f.write(f"POINTS {Nx * Ny * Nz} float\n".encode())

            # Write grid coordinates
            coords = np.zeros((Nx * Ny * Nz, 3), dtype='>f4')

            for k in range(Nz):
                for j in range(Ny):
                    for i in range(Nx):
                        coords[k * Nx * Ny + j * Nx +
                               i] = [self.x[i], self.y[j], self.z[k]]
            coords.tofile(f)

            # Write vector field data
            f.write(f"POINT_DATA {Nx * Ny * Nz}\n".encode())

            # Write vector field data
            for name in var_name_list:
                f.write(f"SCALARS {name} float 1\n".encode())
                f.write(b"LOOKUP_TABLE default\n")
                var = self.get_var(name)
                var_reorder = np.transpose(var, (1, 2, 0))
                field_flat = var_reorder.ravel(order='F').astype('>f4')
                field_flat.tofile(f)

    @classmethod
    def from_toml(cls, config_path):
        config = toml.load(config_path)
        # read parameters from toml file
        nu = config['nu']
        dkx = config['dkx']
        dkz = config['dkz']
        alpha = config['alpha_mesh']
        Nx = config['NX']
        Ny = config['NY']
        Nz = config['NZ']
        return cls(dkz, dkx, alpha, Nz, Nx, Ny, nu)
