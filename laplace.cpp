#include <iostream>
#include <chrono>
#include <cmath>
#include "mpi.h"
#include "cuda_helper.h"

#define TOLERANCE 1.0e-6

void set_bdry(double* x_old_h, int npts_x, int npts_y, int* neighbors);
void exchange_halo(double* x_old_h, int npts_x, int npts_y, int* neighbors, MPI_Datatype column, MPI_Comm comm_2d);
void compute_next(double *x_old_d, double *x_old_h, double *x_new_d, int npts_local_x, int npts_local_y, double *diff_d, double *diff_h, int iter_method);

int main(int argc, char *argv[])
{
    auto start = std::chrono::high_resolution_clock::now();

    int total_procs;
    int my_old_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &total_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_old_rank);
  
    // number of points in x and y directions
    int npts_x, npts_y;

    // number of subdomains in x and y directions
    int ns_x, ns_y;

    // maximum number of iterations
    int max_iter;

    // 0 for Jacobi, 1 for Gauss-Seidel
    int iter_method;

    // get input
    npts_x = atoi(argv[1]);
    npts_y = atoi(argv[2]);
    ns_x = atoi(argv[3]);
    ns_y = atoi(argv[4]);
    max_iter = atoi(argv[5]);
    iter_method = atoi(argv[6]);

    MPI_Comm comm_2d;
    int my_rank;
    int dims[2], periodicity[2], coords[2];
    dims[0] = ns_x;
    dims[1] = ns_y;
    periodicity[0] = 0;
    periodicity[1] = 0;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodicity, 0, &comm_2d);

    if(comm_2d == MPI_COMM_NULL)
    {
        MPI_Finalize();
        return 0;
    }

    MPI_Comm_rank(comm_2d, &my_rank);
    MPI_Cart_coords(comm_2d, my_rank, 2, coords);

    // number of points in x and y directions for each subdomain
    int npts_local_x = npts_x / ns_x + 2;
    if (coords[0] == 0)
    { 
        npts_local_x += npts_x % ns_x;
    }
    int npts_local_y = npts_y / ns_y + 2;
    if (coords[1] == 0)
    {
        npts_local_y += npts_y % ns_y;
    }
    double *x_old_h = new double[npts_local_x * npts_local_y]();
    double *x_new_h = new double[npts_local_x * npts_local_y]();
    double *x_old_d;
    double *x_new_d;
    cudaErrCheck(cudaMalloc(&x_old_d, npts_local_x * npts_local_y * sizeof(double)));
    cudaErrCheck(cudaMalloc(&x_new_d, npts_local_x * npts_local_y * sizeof(double)));

    enum {W, E, S, N};

    int neighbors[4];

    // get neighbor processes
    MPI_Cart_shift(comm_2d, 0, 1, &neighbors[W], &neighbors[E]);
    MPI_Cart_shift(comm_2d, 1, -1, &neighbors[S], &neighbors[N]);

    set_bdry(x_old_h, npts_local_x, npts_local_y, neighbors);

    // MPI datatype for exchange of columns
    MPI_Datatype column;
    MPI_Type_vector(npts_local_y, 1, npts_local_x, MPI_DOUBLE, &column);
    MPI_Type_commit(&column);

    if(my_rank == 0)
    {
        std::cout << "start iteration\n";
    }

    double *diff_h, diff_sum;
    diff_h = new double();
    double *diff_d;
    cudaErrCheck(cudaMalloc(&diff_d, sizeof(double)));
    int steps = 0;
    while(steps < max_iter)
    {
        steps += 1;

        exchange_halo(x_old_h, npts_local_x, npts_local_y, neighbors, column, comm_2d);
        
        // launch kernel
        compute_next(x_old_d, x_old_h, x_new_d, npts_local_x, npts_local_y, diff_d, diff_h, iter_method);
        
        MPI_Allreduce(diff_h, &diff_sum, 1, MPI_DOUBLE, MPI_SUM, comm_2d);

        diff_sum = std::sqrt(diff_sum);

        if(diff_sum < TOLERANCE)
        {
            break;
        }
    }

    if(my_rank == 0)
    {
        std::cout << "end iteration\n";
        std::cout << "iteration steps: " << steps << std::endl;
        std::cout << "diff is " << diff_sum << std::endl;
        std::cout << "start file IO\n";
    }

    // file IO
    MPI_File fh;
    MPI_File_open(comm_2d, "output.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_Offset offset = 0;

    if(coords[1] == 0)
    {
        offset += 0;
    }
    else
    {
        offset += (coords[1] * npts_y / ns_y + npts_y % ns_y) * npts_x * sizeof(double);
    }
    if(coords[0] == 0)
    {
        offset += 0;
    }
    else
    {
        offset += (coords[0] * npts_x / ns_x + npts_x % ns_x) * sizeof(double);
    }

    double *array_ptr = x_old_h;
    for(int i = 1; i < npts_local_y - 1; ++i)
    {
        array_ptr += npts_local_x;
        MPI_File_write_at(fh, offset, array_ptr + 1, npts_local_x - 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
        offset += npts_x * sizeof(double);
    }
    MPI_File_close(&fh);

    if(my_rank == 0)
    {
        std::cout << "end file IO\n";
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if(my_rank == 0)
    {
        std::cout << "Elapsed time: " << elapsed.count() << " s\n";
    }

    delete[] x_old_h;
    delete[] x_new_h;
    delete diff_h;
    cudaFree(x_old_d);
    cudaFree(x_new_d);
    cudaFree(diff_d);
    MPI_Type_free(&column);
    MPI_Finalize();

    return 0;
}