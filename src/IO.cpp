#include "setup.hpp"
#include "toml.hpp"
#include <hdf5.h>
#include <fstream>
#include <vector>
using std::vector;

void load_params(const std::string &filename)
{
    try
    {
        // parse toml file
        auto config = toml::parse(filename);
        dt = toml::find<double>(config, "dt");
        dkx = toml::find<double>(config, "dkx");
        dkz = toml::find<double>(config, "dkz");
        NX = toml::find<size_t>(config, "NX");
        NY = toml::find<size_t>(config, "NY");
        NZ = toml::find<size_t>(config, "NZ");
        alpha_mesh = toml::find<double>(config, "alpha_mesh");
        NT = toml::find<size_t>(config, "NT");
        PX = toml::find<size_t>(config, "PX");
        PZ = toml::find<size_t>(config, "PZ");
        save_interval = toml::find<size_t>(config, "save_interval");
        nu = toml::find<double>(config, "nu");

        NXH = NX / 2;
        NZH = NZ / 2;
        NX2 = NX / 2 * 3;
        NZ2 = NZ / 2 * 3;
        NXHPX = NXH / PX;
        NZPZ = NZ / PZ;
        NYPZ = NY / PZ;
        NZ2PX = NZ2 / PX;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error loading configuration file:" << e.what() << '\n';
    }

    // broadcast parameters to all processors
    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dkx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dkz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NX, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NY, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NZ, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NT, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha_mesh, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&PX, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&PZ, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NXH, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NZH, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NX2, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NZ2, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NXHPX, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NZPZ, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NYPZ, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NZ2PX, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&save_interval, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nu, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // check if all processors have the same parameters
    if (my_rank == 0)
    {
        std::cout << "dkx, dkz = " << dkx << ' ' << dkz << std::endl;
        std::cout << "NX, NY, NZ = " << NX << ' ' << NY << ' ' << NZ << std::endl;
        std::cout << "PX, PZ = " << PX << ' ' << PZ << std::endl;
        std::cout << "NXH, NZH, NX2, NZ2 = " << NXH << ' ' << NZH << ' ' << NX2 << ' ' << NZ2 << std::endl;
        std::cout << "NXHPX, NZPZ = " << NXHPX << ' ' << NZPZ << std::endl;
        std::cout << "nu = " << nu << std::endl;
        std::cout << "dt, NT, save_interval = " << dt << ' ' << NT << ' ' << save_interval << std::endl;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// functions for saving results to file
//////////////////////////////////////////////////////////////////////////////////////////////////////////

void evaluate_array(const Field<Complex> &in,
                    vector<vector<vector<double>>> &out_real,
                    vector<vector<vector<double>>> &out_imag)
{
    size_t nz = in.size_z();
    size_t nx = in.size_x();
    size_t ny = in.size_y();

    for (size_t k = 0; k < nz; k++)
    {
        for (size_t i = 0; i < nx; i++)
        {
            for (size_t j = 0; j < ny; j++)
            {
                out_real[k][i][j] = in(k, i, j).real();
                out_imag[k][i][j] = in(k, i, j).imag();
            }
        }
    }
}

hid_t FILE_CREATE(const std::string &io_filename)
{
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    if (plist_id < 0)
    {
        return -1;
    }

    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    hid_t file_id = H5Fcreate(io_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);

    H5Pclose(plist_id);

    return file_id;
}

void WRITE_3D(hid_t file_id, const std::string &dataset_name, const vector<vector<vector<double>>> &data)
{
    // check if the data is of size NXH x NY x NZ
    if (data.size() != size_t(NZPZ) || data[0].size() != size_t(NXHPX) || data[0][0].size() != size_t(NY))
    {
        std::cout << data.size() << " " << data[0].size() << " " << data[0][0].size() << std::endl;
        std::cerr << "Error: data size is not consistent with NZPZ x NXH x NY  " << std::endl;
        return;
    }

    int dim_rank = 3;
    hsize_t dims[3] = {(hsize_t)NZ, (hsize_t)NXH, (hsize_t)NY};
    hsize_t count[3] = {(hsize_t)NZPZ, (hsize_t)NXHPX, (hsize_t)NY};
    hsize_t offset[3] = {(hsize_t)NZPZ * PZID, (hsize_t)NXHPX * PXID, 0};

    hid_t file_space_id = H5Screate_simple(dim_rank, dims, NULL);
    hid_t dset_id = H5Dcreate(file_id, dataset_name.c_str(), H5T_NATIVE_DOUBLE, file_space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(file_space_id);

    hid_t mem_space_id = H5Screate_simple(dim_rank, count, NULL);
    file_space_id = H5Dget_space(dset_id);
    H5Sselect_hyperslab(file_space_id, H5S_SELECT_SET, offset, NULL, count, NULL);

    // flatten the 3d vector for HDF5 writings
    vector<double> flat_data;
    for (size_t k = 0; k < NZPZ; k++)
        for (size_t i = 0; i < NXHPX; i++)
            for (size_t j = 0; j < NY; j++)
                flat_data.push_back(data[k][i][j]);

    hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
    H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, mem_space_id, file_space_id, plist_id, flat_data.data());

    H5Pclose(plist_id);
    H5Sclose(mem_space_id);
    H5Sclose(file_space_id);
    H5Dclose(dset_id);
}

void WRITE_3D_dealiased(hid_t file_id, const std::string &dataset_name, const Field<double> &data)
{
    // check if the data is of size NZ2PX * NX2 * NYPZ
    if (data.size_x() != size_t(NX2) || data.size_z() != size_t(NZ2PX) || data.size_y() != size_t(NYPZ))
    {
        std::cout << data.size_z() << " " << data.size_x() << " " << data.size_y() << std::endl;
        std::cerr << "Error: data size is not consistent with NZ2PX x NX2 x NYPZ " << std::endl;
        return;
    }

    int dim_rank = 3;
    hsize_t dims[3] = {(hsize_t)NZ2, (hsize_t)NX2, (hsize_t)NY};
    hsize_t count[3] = {(hsize_t)NZ2PX, (hsize_t)NX2, (hsize_t)NYPZ};
    hsize_t offset[3] = {(hsize_t)NZ2PX * PXID, 0, (hsize_t)NYPZ * PZID};

    hid_t file_space_id = H5Screate_simple(dim_rank, dims, NULL);
    hid_t dset_id = H5Dcreate(file_id, dataset_name.c_str(), H5T_NATIVE_DOUBLE, file_space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(file_space_id);

    hid_t mem_space_id = H5Screate_simple(dim_rank, count, NULL);
    file_space_id = H5Dget_space(dset_id);
    H5Sselect_hyperslab(file_space_id, H5S_SELECT_SET, offset, NULL, count, NULL);

    // flatten the 3d vector for HDF5 writings
    vector<double> flat_data;
    for (size_t k = 0; k < NZ2PX; k++)
        for (size_t i = 0; i < NX2; i++)
            for (size_t j = 0; j < NYPZ; j++)
                flat_data.push_back(data(k, i, j));

    hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
    H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, mem_space_id, file_space_id, plist_id, flat_data.data());

    H5Pclose(plist_id);
    H5Sclose(mem_space_id);
    H5Sclose(file_space_id);
    H5Dclose(dset_id);
}

void save_instant_field(const int id_step_)
{

    std::stringstream filename_stream;
    filename_stream << "./DATA/F-" << std::setw(8) << std::setfill('0') << id_step_ << ".H5";
    std::string filename = filename_stream.str();
    // if (my_rank == 0)
    // {
    //     std::cout << "step " << id_step << ", saving data to file:" << filename << std::endl;
    // }

    // IO init
    H5open();

    hid_t file_id = FILE_CREATE(filename);
    if (file_id < 0)
    {
        return;
    }

    vector<vector<vector<double>>> var_real, var_imag;
    var_real.resize(NZPZ, vector<vector<double>>(NXHPX, vector<double>(NY)));
    var_imag.resize(NZPZ, vector<vector<double>>(NXHPX, vector<double>(NY)));

    // save velocity u
    evaluate_array(velx, var_real, var_imag);
    WRITE_3D(file_id, "velx_real", var_real);
    WRITE_3D(file_id, "velx_imag", var_imag);

    // save velocity v
    evaluate_array(vely, var_real, var_imag);
    WRITE_3D(file_id, "vely_real", var_real);
    WRITE_3D(file_id, "vely_imag", var_imag);

    // save velocity w
    evaluate_array(velz, var_real, var_imag);
    WRITE_3D(file_id, "velz_real", var_real);
    WRITE_3D(file_id, "velz_imag", var_imag);

    // // save vorticity omega_x
    // evaluate_array(vorx, var_real, var_imag);
    // WRITE_3D(file_id, "vorx_real", var_real);
    // WRITE_3D(file_id, "vorx_imag", var_imag);

    // // save vorticity omega_y
    // evaluate_array(vory, var_real, var_imag);
    // WRITE_3D(file_id, "vory_real", var_real);
    // WRITE_3D(file_id, "vory_imag", var_imag);

    // // save vorticity omega_z
    // evaluate_array(vorz, var_real, var_imag);
    // WRITE_3D(file_id, "vorz_real", var_real);
    // WRITE_3D(file_id, "vorz_imag", var_imag);

    // // save nonlinear jacx
    // evaluate_array(lmbx, var_real, var_imag);
    // WRITE_3D(file_id, "lmbx_real", var_real);
    // WRITE_3D(file_id, "lmbx_imag", var_imag);

    // // save nonlinear jacy
    // evaluate_array(lmby, var_real, var_imag);
    // WRITE_3D(file_id, "lmby_real", var_real);
    // WRITE_3D(file_id, "lmby_imag", var_imag);

    // // save nonlinear jacz
    // evaluate_array(lmbz, var_real, var_imag);
    // WRITE_3D(file_id, "lmbz_real", var_real);
    // WRITE_3D(file_id, "lmbz_imag", var_imag);

    // // save dealiased field(refined)
    // WRITE_3D_dealiased(file_id, "velx_p", velx_p);
    // WRITE_3D_dealiased(file_id, "vely_p", vely_p);
    // WRITE_3D_dealiased(file_id, "velz_p", velz_p);

    // WRITE_3D_dealiased(file_id, "vorx_p", vorx_p);
    // WRITE_3D_dealiased(file_id, "vory_p", vory_p);
    // WRITE_3D_dealiased(file_id, "vorz_p", vorz_p);

    // WRITE_3D_dealiased(file_id, "lmbx_p", lmbx_p);
    // WRITE_3D_dealiased(file_id, "lmby_p", lmby_p);
    // WRITE_3D_dealiased(file_id, "lmbz_p", lmbz_p);

    // IO finish
    H5Fclose(file_id);
    H5close();
}
