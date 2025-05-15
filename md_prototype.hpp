#ifndef MD_PROTOTYPE_HPP
#define MD_PROTOTYPE_HPP

#include <mpi.h>
#include <vector>
#include <chrono>
#include <thread>
#include "mpi_helper.hpp"

class CoordReceiverAlgorithm {
public:
    CoordReceiverAlgorithm(MPI_Comm comm, int rank, int partner_rank);
    void receive_data(int iter);
    void receive_ctrl_msg(int iter);
    void run(int iter);

private:
    MPI_Comm comm_;
    int rank_;
    int partner_rank_;
    std::vector<float> data_;  // 15x3 matrix flattened
    std::vector<int> ctrldata_;
};

// Helper function to calculate energies (simulated delay)
void calculate_energies(float iter_delay);

#endif // MD_PROTOTYPE_HPP 