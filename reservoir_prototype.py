from mpi4py import MPI
import mdtraj as md
import numpy as np
import time
from config_prototype import yaml_to_object
from mpi_helper import MessageType, CtrlMsg, sync_process


class TrrReader:
    def __init__(self, trr, gro):
        # Load the trajectory with the topology file
        traj = md.load_trr(trr, top=gro)
        # Number of frames
        print("Number of frames:", traj.n_frames)
        # Box size (unit cell vectors)
        print("Box vectors shape:", traj.unitcell_vectors.shape)
        # Shape of coordinates
        print("Coordinates shape:", traj.xyz.shape)
        # coordinates of coordinates
        print(
            f"Coordinates: type={type(traj.xyz[0])} shape={traj.xyz[0].shape} data_type={traj.xyz[0].dtype}"
        )
        self.traj = traj

    def get_frame_coords(self, frame_index):
        if 0 <= frame_index and frame_index < self.traj.n_frames:
            return self.traj.xyz[frame_index], self.traj.unitcell_vectors[frame_index]
        raise ValueError(f"frame_index beyond the range [0,{self.traj.n_frames})")

    def get_num_frames(self):
        return self.traj.n_frames


def main():
    # config = ConfigPrototype()
    config = yaml_to_object()
    sending_interval = config.reservoir_sending_interval
    num_iter = config.reservoir_num_iter
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    partner_rank = sync_process(config.reservoir_lambda_values, "S", comm)
    print(f"Sender with rank {rank} started")
    if partner_rank != -1:
        msg = np.array([CtrlMsg.eStart.value], dtype=np.int32)
        comm.Send(msg, dest=partner_rank, tag=MessageType.eCtrlMsg.value)
        trr_reader,box = TrrReader(config.reservoir_trr_file, config.reservoir_gro_file)
        for i in range(num_iter):
            time.sleep(config.reservoir_sending_interval)  # Small delay between sends
            data = trr_reader.get_frame_coords(
                np.random.randint(0, trr_reader.get_num_frames())
            )
            print(
                f"Reservoir with rank {rank} sending data with shape {data.shape} (count: {i + 1}/{num_iter})"
            )
            comm.Send(box, dest=partner_rank, tag=MessageType.eDataMesg.value)
            comm.Send(data, dest=partner_rank, tag=MessageType.eDataMesg.value)

        time.sleep(sending_interval)
        msg = np.array([CtrlMsg.eStart.value], dtype=np.int32)
        comm.Send(msg, dest=partner_rank, tag=MessageType.eCtrlMsg.value)
    print(f"Sender with rank {rank} finished")


main()
