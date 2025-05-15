from mpi4py import MPI
import numpy as np
import time
from config_prototype import yaml_to_object
from mpi_helper import MessageType,CtrlMsg,sync_process

class CoordReceiverAlgorithm:  

    def __init__(self, comm, rank, partner_rank):
        self.comm = comm
        self.rank = rank
        self.partner_rank = partner_rank
        # Create a numpy array to receive into (like int in C)
        # data = np.empty(1, dtype='i')  # 'i' = 32-bit int
        self.data = np.zeros((15, 3), dtype=np.float32)
        self.ctrldata = np.zeros(1, dtype=np.int32)
    def receive_data(self,  iter):
        flag = self.comm.Iprobe(source=self.partner_rank, tag=MessageType.eDataMesg.value)
        if flag:
            self.comm.Recv(self.data, source=self.partner_rank, tag=MessageType.eDataMesg.value)
            print(
                f"Md Pythoncode with rank {self.rank} received data with shape : {self.data.shape} (md iterration: {iter})"
            )
        #energy1, energy2 = calclulate_energies(data)

    def receive_ctrl_msg(self, iter):
        flag = self.comm.Iprobe(source=self.partner_rank, tag=MessageType.eCtrlMsg.value)
        if flag:
            self.comm.Recv(self.ctrldata, source=self.partner_rank, tag=MessageType.eCtrlMsg.value)
            print(
                f"Md code with rank {self.rank} received ctrl msg : {self.ctrldata} (iterration count: {iter})"
            )

    def run(self, iter):
        if self.partner_rank != -1:
            self.receive_data(iter)
            self.receive_ctrl_msg(iter)
        
            

def calculate_energies(iter_delay):
    time.sleep(iter_delay)

def main():
    #config = ConfigPrototype()
    config = yaml_to_object()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print(f"Receiver with rank {rank} started")
    partner_rank = sync_process(config.md_lambda_values, 'R', comm)
    receiver = CoordReceiverAlgorithm(comm, rank, partner_rank)
    for iter in range(config.md_num_iter):
        calculate_energies(config.md_iter_delay)
        receiver.run(iter)
            
    print(f"Receiver with rank {rank} finished")

main()
