#include <iostream>
#include <iomanip>
//#include <yaml-cpp/yaml.h>
#include "md_prototype.hpp"
#include "yaml/Yaml.hpp"

CoordReceiverAlgorithm::CoordReceiverAlgorithm(MPI_Comm comm, int rank, int partner_rank)
    : comm_(comm), rank_(rank), partner_rank_(partner_rank)
{
    // Initialize data_ as 15x3 matrix (flattened)
    data_.resize(15 * 3, 0.0f);
    ctrldata_.resize(1, 0);
}

void CoordReceiverAlgorithm::receive_data(int iter)
{
    int flag;
    MPI_Status status;

    MPI_Iprobe(partner_rank_, static_cast<int>(MessageType::eDataMesg), comm_, &flag, &status);
    if (flag)
    {
        MPI_Recv(data_.data(), data_.size(), MPI_FLOAT,
                 partner_rank_, static_cast<int>(MessageType::eDataMesg),
                 comm_, &status);

        std::cout << "Md C++ code with rank " << rank_
                  << " received data with shape : [15, 3] (md iteration: "
                  << iter << ")" << std::endl;
    }
}

void CoordReceiverAlgorithm::receive_ctrl_msg(int iter)
{
    int flag;
    MPI_Status status;

    MPI_Iprobe(partner_rank_, static_cast<int>(MessageType::eCtrlMsg), comm_, &flag, &status);
    if (flag)
    {
        MPI_Recv(ctrldata_.data(), ctrldata_.size(), MPI_INT,
                 partner_rank_, static_cast<int>(MessageType::eCtrlMsg),
                 comm_, &status);

        std::cout << "Md C++ code with rank " << rank_
                  << " received ctrl msg : " << ctrldata_[0]
                  << " (iteration count: " << iter << ")" << std::endl;
    }
}

void CoordReceiverAlgorithm::run(int iter)
{
    if (partner_rank_ != -1)
    {
        receive_data(iter);
        receive_ctrl_msg(iter);
    }
}

void calculate_energies(float iter_delay)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(
        static_cast<int>(iter_delay)));
}

// Configuration structure to hold parameters
struct Config
{
    std::vector<std::string> md_lambda_values;
    int md_num_iter;
    float md_iter_delay;
};

std::vector<float> parseFloatVector(const std::string& str) {
    // Check for valid input format: starts with '[' and ends with ']'
    if (str.empty() || str.front() != '[' || str.back() != ']') {
        throw std::invalid_argument("Input string must start with '[' and end with ']'");
    }

    std::vector<float> vec;
    std::stringstream ss(str.substr(1, str.size() - 2));  // Remove square brackets and use stringstream

    for (std::string temp; std::getline(ss, temp, ','); ) {
        // Convert each trimmed string to float and add to vector
        vec.push_back(std::stof(temp.erase(0, temp.find_first_not_of(" \t")).erase(temp.find_last_not_of(" \t") + 1)));
    }

    return vec;
}

class ConfigPrototype {
public:
    ConfigPrototype(const std::string& file_name = "config.yaml")
    {
        /*YAML::Node config = YAML::LoadFile(file_name);
        md_num_iter = config["md_num_iter"].as<int>(3);
        md_iter_delay = config["receiving_interval"].as<float>(0.1)*1000;
        md_lambda_values = config["md_lambda_values"].as<std::vector<float>>(std::vector<float>());
        */
        Yaml::Node root;
        Yaml::Parse(root, file_name.c_str());
        md_num_iter = root["md_num_iter"].As<int>();
        md_iter_delay = root["md_iter_delay"].As<float>();
        md_lambda_values = parseFloatVector(root["md_lambda_values"].As<std::string>());
        /*
        if (lamdas.IsSequence())
        {
            for (auto i = 0 ; i != lamdas.Size();i++)
            {
                md_lambda_values.push_back(lamdas[i].As<float>());
            }
        }
        */
        md_num_iter = 3;
        md_iter_delay = 0.1 * 1000;
        md_lambda_values = {0,0.1,0.2,0.4};
    }
public:    
    int md_num_iter;
    float md_iter_delay;
    std::vector<float> md_lambda_values;

};

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    std::cout << "Receiver with rank " << rank << " started" << std::endl;

    auto config = ConfigPrototype();
    int partner_rank = sync_process(config.md_lambda_values, 'R', comm);

    CoordReceiverAlgorithm receiver(comm, rank, partner_rank);

    for (int iter = 0; iter < config.md_num_iter; ++iter)
    {
        calculate_energies(config.md_iter_delay);
        receiver.run(iter);
    }

    std::cout << "Receiver with rank " << rank << " finished" << std::endl;

    MPI_Finalize();
    return 0;
}