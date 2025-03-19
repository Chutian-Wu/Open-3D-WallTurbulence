#include "setup.hpp"
#include <cassert>

void EXCHANGE_Y2Z(const Field<Complex> &var_x1z1_py, Field<Complex> &var_x1z1_pz)
{

    // check var_x1z1_py.shape()==[NZPZ][NXHPX][NY]
    assert(var_x1z1_py.size_z() == size_t(NZPZ));
    assert(var_x1z1_py.size_x() == size_t(NXHPX));
    assert(var_x1z1_py.size_y() == size_t(NY));

    // check var_x1z1_pz.shape()==[NZ][NXHPX][NYPZ]
    assert(var_x1z1_pz.size_z() == size_t(NZ));
    assert(var_x1z1_pz.size_x() == size_t(NXHPX));
    assert(var_x1z1_pz.size_y() == size_t(NYPZ));

    size_t all2all_size = NXHPX * NYPZ * NZPZ;
    static thread_local std::vector<Complex> send(all2all_size * PZ);
    static thread_local std::vector<Complex> recv(all2all_size * PZ);

    // pack data into send buffer
    size_t idx = 0;
    for (size_t p = 0; p < PZ; p++)
    {
        for (size_t k = 0; k < NZPZ; k++)
        {
            for (size_t i = 0; i < NXHPX; i++)
            {
                for (size_t j = 0; j < NYPZ; j++)
                {
                    size_t y_idx = p * NYPZ + j;
                    send[idx++] = var_x1z1_py(k, i, y_idx);
                }
            }
        }
    }

    // MPI send/recv
    int comm_size;
    MPI_Comm_size(MPI_COMM_X, &comm_size);
    if (comm_size != int(PZ))
    {
        throw std::runtime_error("MPI_COMM_X size does not match PZ!");
    }
    MPI_Alltoall(send.data(), all2all_size, MPI_DOUBLE_COMPLEX,
                 recv.data(), all2all_size, MPI_DOUBLE_COMPLEX,
                 MPI_COMM_X);

    // unpack data from recv buffer, loop order should be the same as packing
    idx = 0;
    for (size_t p = 0; p < PZ; p++)
    {
        for (size_t k = 0; k < NZPZ; k++)
        {
            for (size_t i = 0; i < NXHPX; i++)
            {
                for (size_t j = 0; j < NYPZ; j++)
                {
                    size_t z_idx = p * NZPZ + k;
                    var_x1z1_pz(z_idx, i, j) = recv[idx++];
                }
            }
        }
    }
}

void EXCHANGE_Z2X(const Field<Complex> &var_x1z2_pz, Field<Complex> &var_x1z2_px)
{
    // This function changes parallel direction from Z to X
    // MPI_Barrier(MPI_COMM_Z);
    // double start_time = MPI_Wtime();

    // check var_x1z2_pz.shape()==[NZ2][NXHPX][NYPZ]
    assert(var_x1z2_pz.size_z() == size_t(NZ2));
    assert(var_x1z2_pz.size_x() == size_t(NXHPX));
    assert(var_x1z2_pz.size_y() == size_t(NYPZ));

    // check var_x1z2_px.shape()==[NZ2PX][NXH][NYPZ]
    assert(var_x1z2_px.size_z() == size_t(NZ2PX));
    assert(var_x1z2_px.size_x() == size_t(NXH));
    assert(var_x1z2_px.size_y() == size_t(NYPZ));

    size_t all2all_size = NXHPX * NYPZ * NZ2PX;

    //////////////////////////////////////////////////////////////////////
    /* The first way */
    static thread_local std::vector<Complex> send(all2all_size * PX);
    static thread_local std::vector<Complex> recv(all2all_size * PX);

    // copy data from var_x1z1_py to send buffer
    size_t idx = 0;
    for (size_t p = 0; p < PX; p++)
    {
        for (size_t k = 0; k < NZ2PX; k++)
        {
            for (size_t i = 0; i < NXHPX; i++)
            {
                for (size_t j = 0; j < NYPZ; j++)
                { // global y index
                    size_t z_idx = p * NZ2PX + k;
                    send[idx++] = var_x1z2_pz(z_idx, i, j);
                }
            }
        }
    }

    // MPI send/recv
    int comm_size;
    MPI_Comm_size(MPI_COMM_Z, &comm_size);
    if (comm_size != int(PX))
    {
        throw std::runtime_error("MPI_COMM_Z size does not match PX!");
    }
    MPI_Alltoall(send.data(), all2all_size, MPI_DOUBLE_COMPLEX,
                 recv.data(), all2all_size, MPI_DOUBLE_COMPLEX,
                 MPI_COMM_Z);

    // recover data from recv buffer to var_z
    idx = 0;
    for (size_t p = 0; p < PX; p++)
    {
        for (size_t k = 0; k < NZ2PX; k++)
        {
            for (size_t i = 0; i < NXHPX; i++)
            {
                for (size_t j = 0; j < NYPZ; j++)
                {
                    size_t x_idx = p * NXHPX + i;
                    var_x1z2_px(k, x_idx, j) = recv[idx++];
                }
            }
        }
    }
}

void EXCHANGE_X2Z(const Field<Complex> &var_x1z2_px, Field<Complex> &var_x1z2_pz)
{
    // This function changes parallel direction from X to Z
    // MPI_Barrier(MPI_COMM_Z);
    // double start_time = MPI_Wtime();

    // double start_time = MPI_Wtime();

    // check var_x1z2_px.shape()==[NZ2PX][NXH][NYPZ]
    assert(var_x1z2_px.size_z() == size_t(NZ2PX));
    assert(var_x1z2_px.size_x() == size_t(NXH));
    assert(var_x1z2_px.size_y() == size_t(NYPZ));

    // check var_x1z2_pz.shape()==[NZ2][NXHPX][NYPZ]
    assert(var_x1z2_pz.size_z() == size_t(NZ2));
    assert(var_x1z2_pz.size_x() == size_t(NXHPX));
    assert(var_x1z2_pz.size_y() == size_t(NYPZ));

    size_t all2all_size = NZ2PX * NXHPX * NYPZ;

    //////////////////////////////////////////////////////////////////////
    /* The first way */
    static thread_local std::vector<Complex> send(all2all_size * PX);
    static thread_local std::vector<Complex> recv(all2all_size * PX);

    // copy data from var_x1z1_py to send buffer
    size_t idx = 0;
    for (size_t p = 0; p < PX; p++)
    {
        for (size_t k = 0; k < NZ2PX; k++)
        {
            for (size_t i = 0; i < NXHPX; i++)
            {
                for (size_t j = 0; j < NYPZ; j++)
                {
                    // global x index
                    size_t x_idx = p * NXHPX + i;
                    send[idx++] = var_x1z2_px(k, x_idx, j);
                }
            }
        }
    }

    // MPI send/recv
    int comm_size;
    MPI_Comm_size(MPI_COMM_Z, &comm_size);
    if (comm_size != int(PX))
    {
        throw std::runtime_error("MPI_COMM_Z size does not match PX!");
    }
    MPI_Alltoall(send.data(), all2all_size, MPI_DOUBLE_COMPLEX,
                 recv.data(), all2all_size, MPI_DOUBLE_COMPLEX,
                 MPI_COMM_Z);

    // recover data from recv buffer to var_z
    idx = 0;
    for (size_t p = 0; p < PX; p++)
    {
        for (size_t k = 0; k < NZ2PX; k++)
        {
            for (size_t i = 0; i < NXHPX; i++)
            {
                for (size_t j = 0; j < NYPZ; j++)
                {
                    size_t z_idx = p * NZ2PX + k;
                    var_x1z2_pz(z_idx, i, j) = recv[idx++];
                }
            }
        }
    }

    // double end_time = MPI_Wtime();
    // double elapsed_time = end_time - start_time;
    // if (my_rank == 0)
    // {
    //     std::cout << "using time " << elapsed_time << "\n";
    // }
}

void EXCHANGE_Z2Y(const Field<Complex> &var_x1z1_pz, Field<Complex> &var_x1z1_py)
{

    // This function changes parallel direction from Z to Y
    // MPI_Barrier(MPI_COMM_X);
    // double start_time = MPI_Wtime();

    // check var_x1z1_pz.shape()==[NZ][NXHPX][NYPZ]
    assert(var_x1z1_pz.size_z() == size_t(NZ));
    assert(var_x1z1_pz.size_x() == size_t(NXHPX));
    assert(var_x1z1_pz.size_y() == size_t(NYPZ));

    // check var_x1z1_py.shape()==[NZPZ][NXHPX][NY]
    assert(var_x1z1_py.size_z() == size_t(NZPZ));
    assert(var_x1z1_py.size_x() == size_t(NXHPX));
    assert(var_x1z1_py.size_y() == size_t(NY));

    size_t all2all_size = NXHPX * NYPZ * NZPZ;

    //////////////////////////////////////////////////////////////////////
    /* The first way */
    static thread_local std::vector<Complex> send(all2all_size * PZ);
    static thread_local std::vector<Complex> recv(all2all_size * PZ);

    // copy data from var_x1z1_py to send buffer
    size_t idx = 0;
    for (size_t p = 0; p < PZ; p++)
    {
        for (size_t k = 0; k < NZPZ; k++)
        {
            // global z index
            size_t z_idx = p * NZPZ + k;
            for (size_t i = 0; i < NXHPX; i++)
            {
                for (size_t j = 0; j < NYPZ; j++)
                {
                    send[idx++] = var_x1z1_pz(z_idx, i, j);
                }
            }
        }
    }

    // MPI send/recv
    int comm_size;
    MPI_Comm_size(MPI_COMM_X, &comm_size);
    if (comm_size != int(PZ))
    {
        throw std::runtime_error("MPI_COMM_X size does not match PZ!");
    }

    MPI_Alltoall(send.data(), all2all_size, MPI_DOUBLE_COMPLEX,
                 recv.data(), all2all_size, MPI_DOUBLE_COMPLEX,
                 MPI_COMM_X);

    // recover data from recv buffer to var_z
    idx = 0;
    for (size_t p = 0; p < PZ; p++)
    {
        for (size_t k = 0; k < NZPZ; k++)
        {
            for (size_t i = 0; i < NXHPX; i++)
            {
                for (size_t j = 0; j < NYPZ; j++)
                {
                    size_t y_idx = p * NYPZ + j;
                    var_x1z1_py(k, i, y_idx) = recv[idx++];
                }
            }
        }
    }
}
