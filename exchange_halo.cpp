#include "mpi.h"

void exchange_halo(double* x_old_h, int npts_x, int npts_y, int* neighbors, MPI_Datatype column, MPI_Comm comm_2d)
{
    enum {W, E, S, N};
    // Send row to NORTH neighbor, and receive row from SOUTH neighbor
    MPI_Sendrecv(&x_old_h[npts_x], npts_x, MPI_DOUBLE, neighbors[N], 0, &x_old_h[(npts_y - 1) * npts_x], npts_x, MPI_DOUBLE, neighbors[S], 0, comm_2d, MPI_STATUS_IGNORE);
    // Send row to SOUTH neighbor, and receive row from NORTH neighbor
    MPI_Sendrecv(&x_old_h[(npts_y - 2) * npts_x], npts_x, MPI_DOUBLE, neighbors[S], 0, &x_old_h[0], npts_x, MPI_DOUBLE, neighbors[N], 0, comm_2d, MPI_STATUS_IGNORE);
    // Send column to WEST neighbor, and receive column from EAST neighbor
    MPI_Sendrecv(&x_old_h[1], 1, column, neighbors[W], 1, &x_old_h[npts_x - 1], 1, column, neighbors[E], 1, comm_2d, MPI_STATUS_IGNORE);
    // Send column to EAST neighbor, and receive column from WEST neighbor
    MPI_Sendrecv(&x_old_h[npts_x - 2], 1, column, neighbors[E], 1, &x_old_h[0], 1, column, neighbors[W], 1, comm_2d, MPI_STATUS_IGNORE);
}