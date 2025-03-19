#ifndef SETUP_HPP
#define SETUP_HPP

#include <iostream>
#include <hdf5.h>
#include <fftw3.h>
#include <mpi.h>
#include <filesystem>
#include <complex>
#include <cmath>
#include <string>
#include <vector>
#include "linalg.hpp"
using Complex = std::complex<double>;
using std::vector;

#define PI 3.141592653589793238462643383279502884197
#define EPSILON 1e-16
#define IUNIT std::complex<double>(0.0, 1.0)

// mesh size and parallel decomposition
inline double dt;
inline double dkx, dkz;
inline size_t NX, NY, NZ;
inline double alpha_mesh;
inline size_t NXH, NZH;
inline size_t NX2, NZ2;
inline size_t PX, PZ;
inline size_t NXHPX, NZPZ, NYPZ;
inline size_t NX2PX, NZ2PZ, NZ2PX;

// IO
inline size_t NT, save_interval;
inline size_t id_step = 0;

// physical parameters
inline double nu; // kinetic viscosity

// MPI
inline int my_rank, numProcs;
inline int PZID, PXID;
inline MPI_Comm MPI_COMM_X;
inline MPI_Comm MPI_COMM_Z;

// wave numbers and y mesh
inline Vec<double> kx_local;
inline Vec<double> kz_local;
inline Vec<double> y, s;
inline double ds;
inline Vec<double> dyds1, dyds2, dyds3, dyds4;
inline Vec<double> c11;
inline Vec<double> c21, c22;
inline Vec<double> c31, c32, c33;
inline Vec<double> c41, c42, c43, c44;
inline Mat DY1, DY2, DY3, DY4;

// variables
inline Field<Complex> velx, vely, velz; // velocity components in wave space
inline Field<Complex> vorx, vory, vorz; // vorticity components in wave space
inline Field<Complex> lmbx, lmby, lmbz; // Lamb vector components in wave space

inline Field<Complex> Hv, Hg; // right-hand (source terms) of equations for wall-normal components

inline Field<Complex> vely_0, vely_1, vely_2; // velocity fields used to integrate time
inline Field<Complex> vory_0, vory_1, vory_2; // vorticity fields used to integrate time
inline Field<Complex> Hv_0, Hv_1, Hv_2;
inline Field<Complex> Hg_0, Hg_1, Hg_2;
inline Vec<Complex> lmbx_mean_0, lmbx_mean_1, lmbx_mean_2; // vars used for mean velocity
inline Vec<Complex> lmbz_mean_0, lmbz_mean_1, lmbz_mean_2; // vars used for mean velocity
inline Vec<Complex> velx_mean_0, velx_mean_1, velx_mean_2; // vars used for mean velocity
inline Vec<Complex> velz_mean_0, velz_mean_1, velz_mean_2; // vars used for mean velocity

inline Field<double> velx_p, vely_p, velz_p; // velocity components in physical space (3/2 refined mesh)
inline Field<double> vorx_p, vory_p, vorz_p; // vorticuty components in physical space (3/2 refined mesh)
inline Field<double> lmbx_p, lmby_p, lmbz_p; // Lamb vector components in physical space (3/2 refined mesh)

inline double fx_fric, fz_fric;                // friction force
inline double fx_fric_0, fx_fric_1, fx_fric_2; // friction force of former steps
inline double fz_fric_0, fz_fric_1, fz_fric_2; // friction force of former steps

// FFT stuff
inline fftw_complex *fft_temp_x1, *fft_temp_x2, *fft_temp_z1, *fft_temp_z2;
inline fftw_plan plan_xf, plan_xb, plan_zf, plan_zb;
void init_fft();
void cleanup_fft();
// MPI stuff
void EXCHANGE_Y2Z(const Field<Complex> &var_x1z1_py, Field<Complex> &var_x1z1_pz);
void EXCHANGE_Z2X(const Field<Complex> &var_x1z2_pz, Field<Complex> &var_x1z2_px);
void EXCHANGE_X2Z(const Field<Complex> &var_x1z2_px, Field<Complex> &var_x1z2_pz);
void EXCHANGE_Z2Y(const Field<Complex> &var_x1z1_pz, Field<Complex> &var_x1z1_py);
// functions
// function to load the params from .toml file
void load_params(const std::string &filename);
// function to allocate the complex arrays
void allocate_var();

// functions to initialize the fields
void init_fields();
// functions to initialize the matrices used for wall-normal discretization and solve
void init_mat();

// functions to get the Lamb vector
void get_vor_from_vel();
void solve_horizontal();
void solve_mean();
void solve_friction_force();
void Lamb_vec();
void time_Integration();
void nonlinear_source();
void Fourier2Physical_dealiased(const Field<Complex> &u_, Field<double> &up_);
void Physical2Fourier_dealiased(const Field<double> &up_, Field<Complex> &u_);

// function to save results
void save_instant_field(const int id_step);

#endif