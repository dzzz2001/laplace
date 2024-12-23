void set_bdry(double* x_old_h, int npts_x, int npts_y, int* neighbors)
{
    // West
    if (neighbors[0] < 0)
        for (int i = 0; i < npts_y; ++i)
        {
            x_old_h[i * npts_x] = x_old_h[i * npts_x + 1] = 1.0;
        }
    // East
    if (neighbors[1] < 0)
        for (int i = 0; i < npts_y; ++i)
        {
            x_old_h[i * npts_x + npts_x - 2] = x_old_h[i * npts_x + npts_x - 1] = 1.0;
        }
    // South
    if (neighbors[2] < 0)
        for (int i = 0; i < npts_x; ++i)
        {
            x_old_h[(npts_y - 2) * npts_x + i] = x_old_h[(npts_y - 1) * npts_x + i] = 1.0;
        }
    // North
    if (neighbors[3] < 0)
        for (int i = 0; i < npts_x; ++i)
        {
            x_old_h[i] = x_old_h[npts_x + i] = 1.0;
        }
}