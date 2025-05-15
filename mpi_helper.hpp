#ifndef MPI_HELPER_HPP
#define MPI_HELPER_HPP

#include <mpi.h>
#include <vector>
#include <string>
#include <set>
#include <stdexcept>
#include <iostream>

// Message type enum
enum class MessageType
{
    eDataMesg = 123,
    eCtrlMsg = 456
};

// Control message enum
enum class CtrlMsg
{
    eStart = 1,
    eEnd = 2
};

// Helper function to get splitted rank
int get_splitted_rank(const std::vector<std::string::value_type> &all_participants_type, int rank, const std::string::value_type &type)
{
    int splitted_rank = 0;
    for (int r = 0; r < all_participants_type.size(); ++r)
    {
        if (r == rank)
        {
            break;
        }
        if (all_participants_type[r] == type)
        {
            splitted_rank++;
        }
    }
    return splitted_rank;
}

// Helper function to find partner rank based on keys
int find_keys_partner(const std::vector<float> &all_participants_keys, int rank)
{
    for (int i = 0; i < all_participants_keys.size(); ++i)
    {
        if (i == rank)
        {
            continue;
        }
        if (all_participants_keys[i] == all_participants_keys[rank])
        {
            return i;
        }
    }
    return -1;
}

// Helper function to validate lambda values
void validate_lambda_values(const std::vector<float> &all_participants_keys,
                            const std::vector<std::string::value_type> &all_participants_type)
{
    std::vector<float> sender_lambdas;
    std::vector<float> receiver_lambdas;

    for (size_t i = 0; i < all_participants_type.size(); ++i)
    {
        if (all_participants_type[i] == 'S')
        {
            sender_lambdas.push_back(all_participants_keys[i]);
        }
        else if (all_participants_type[i] == 'R')
        {
            receiver_lambdas.push_back(all_participants_keys[i]);
        }
    }

    std::set<float> sender_set(sender_lambdas.begin(), sender_lambdas.end());
    std::set<float> receiver_set(receiver_lambdas.begin(), receiver_lambdas.end());

    std::set<float> missing_values;
    for (const auto &val : sender_set)
    {
        if (receiver_set.find(val) == receiver_set.end())
        {
            missing_values.insert(val);
        }
    }

    if (!missing_values.empty())
    {
        std::string error_msg = "Following values from sender_lambdas are missing in receiver_lambdas: ";
        for (const auto &val : missing_values)
        {
            error_msg += std::to_string(val) + " ";
        }
        throw std::runtime_error(error_msg);
    }
}

// Main synchronization function
int sync_process(const std::vector<float> &keys, const std::string::value_type &type, MPI_Comm comm)
{
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    // Gather all participant types
    std::vector<std::string::value_type> all_participants_type(size);
    MPI_Allgather(&type, 1, MPI_CHAR,
                  all_participants_type.data(), 1, MPI_CHAR, comm);

    if (rank == 0)
    {
        for (const auto &t : all_participants_type)
        {
            std::cout << t << " ";
        }
        std::cout << std::endl;
    }

    // Count processes of specific type
    int num_type_processes = 0;
    for (const auto &t : all_participants_type)
    {
        if (t == type)
        {
            num_type_processes++;
        }
    }

    if (num_type_processes != keys.size())
    {
        // std::string error_msg = "Number of " + std::string(1, type) + " processes is not equal to the number of " + std::string(1, type) + " keys";
        if (type == 'S')
        {
            throw std::runtime_error("Number of sender processes is not equal to the number of sender keys size: " + std::to_string(keys.size()) + " != " + std::to_string(num_type_processes));
        }
        else if (type == 'R')
        {
            throw std::runtime_error("Number of receiver processes is not equal to the number of receiver keys size: " + std::to_string(keys.size()) + " != " + std::to_string(num_type_processes));
        }
    }

    // Get splitted rank and gather
    int splitted_rank = get_splitted_rank(all_participants_type, rank, type);
    std::vector<int> all_participants_splitted_rank(size);
    MPI_Allgather(&splitted_rank, 1, MPI_INT,
                  all_participants_splitted_rank.data(), 1, MPI_INT, comm);

    if (rank == 0)
    {
        for (const auto &r : all_participants_splitted_rank)
        {
            std::cout << r << " ";
        }
        std::cout << std::endl;
    }

    // Gather all participants keys
    std::vector<float> all_participants_keys(size);
    float local_key = keys[splitted_rank];
    MPI_Allgather(&local_key, 1, MPI_FLOAT,
                  all_participants_keys.data(), 1, MPI_FLOAT, comm);

    if (rank == 0)
    {
        for (const auto &k : all_participants_keys)
        {
            std::cout << k << " ";
        }
        std::cout << std::endl;
    }

    validate_lambda_values(all_participants_keys, all_participants_type);

    int partner_rank = find_keys_partner(all_participants_keys, rank);
    std::vector<int> all_participants_partners_rank(size);
    MPI_Allgather(&partner_rank, 1, MPI_INT,
                  all_participants_partners_rank.data(), 1, MPI_INT, comm);

    if (rank == 0)
    {
        for (const auto &p : all_participants_partners_rank)
        {
            std::cout << p << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Partner rank for rank " << rank << " is " << partner_rank << std::endl;
    return partner_rank;
}

#endif // MPI_HELPER_HPP