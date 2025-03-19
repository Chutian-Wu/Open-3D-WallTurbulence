#include <iostream>
#include <filesystem>
#include "setup.hpp"

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    /*Init MPI*/
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /*Load configuration file*/
    if (argc < 2)
    {
        if (my_rank == 0)
        {
            std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    std::string filename = argv[1];
    load_params(filename);

    MPI_Barrier(MPI_COMM_WORLD);

    /*MPI domain decomposition*/
    PZID = my_rank / int(PX);
    PXID = my_rank - PZID * int(PX);

    std::cout << "rank " << my_rank << " PZID " << PZID << " PXID " << PXID << "\n";
    int new_rank, new_size;

    MPI_Comm_split(MPI_COMM_WORLD, PXID, PZID, &MPI_COMM_X);
    MPI_Comm_rank(MPI_COMM_X, &new_rank);
    MPI_Comm_size(MPI_COMM_X, &new_size);

    if (my_rank == 0)
    {
        std::cout << "size of MPI_COMM_X " << new_size << "\n";
    }

    MPI_Comm_split(MPI_COMM_WORLD, PZID, PXID, &MPI_COMM_Z);
    MPI_Comm_rank(MPI_COMM_Z, &new_rank);
    MPI_Comm_size(MPI_COMM_Z, &new_size);

    if (my_rank == 0)
    {
        std::cout << "size of MPI_COMM_Z " << new_size << "\n";
    }

    /*create essential folders*/
    if (my_rank == 0)
    {
        if (!fs::exists("DATA"))
        {
            fs::create_directory("DATA");
        }
    }

    allocate_var(); /*Allocate arrays*/
    init_mat();     /*preprocess of linear system*/
    init_fft();     /*preprocess of FFT*/
    init_fields();  /*Init fields*/

    /*Time integration*/
    for (size_t i = 0; i <= NT; i++)
    {

        if (id_step % save_interval == 0)
        {
            save_instant_field(id_step);
        }

        double time_start = MPI_Wtime();
        time_Integration();
        double time_end = MPI_Wtime();

        if (my_rank == 0)
        {
            std::cout << "time step, " << id_step << ", using time, " << time_end - time_start << " ";
        }

        id_step += 1;
    }

    /*Finalize MPI*/
    MPI_Finalize();
    cleanup_fft();
    return 0;
}