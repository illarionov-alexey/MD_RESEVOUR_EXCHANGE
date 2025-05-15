from mpi4py import MPI
from enum import Enum
import numpy as np

class MessageType(Enum):
    eDataMesg = 123
    eCtrlMsg = 456

class CtrlMsg(Enum):
    eStart = 1
    eEnd = 2

def get_splitted_rank(all_participants_type, rank, type):
    splitted_rank = 0
    for r, t in enumerate(all_participants_type):
        if r == rank:
            break
        if t == type:
            splitted_rank += 1
    return splitted_rank

def find_keys_partner(all_participants_keys, rank):
    for i, key in enumerate(all_participants_keys):
        if i == rank:
            continue
        if key == all_participants_keys[rank]:
            return i
    return -1

def validate_lambda_values(all_participants_keys, all_participants_type):
    sender_lamdas = [all_participants_keys[i] for i,x in enumerate(all_participants_type) if x == 'S']
    receiver_lamdas = [all_participants_keys[i] for i,x in enumerate(all_participants_type) if x == 'R']
    # Validate that all sender_lamdas values exist in receiver_lamdas
    missing_values = set(sender_lamdas) - set(receiver_lamdas)
    if missing_values:
        raise ValueError(
            f"Following values from sender_lamdas are missing in receiver_lamdas: {missing_values}"
        )

def sync_process(keys, type, comm):
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    char_array = np.array([ord(type)], dtype='b')  # 'S' = 83
    recv_array = np.empty(size, dtype='b')
    # Use MPI.Allgather to gather raw characters
    comm.Allgather([char_array, MPI.CHAR], [recv_array, MPI.CHAR])
    all_participants_type = [chr(x) for x in recv_array]
    if rank == 0:
        print(all_participants_type)

    num_type_processes = len([x for x in all_participants_type if x == type])
    if num_type_processes != len(keys):
        if type == "S":
            raise ValueError(
                f"Number of sender processes is not equal to the number of sender keys"
            )
        elif type == "R":
            raise ValueError(
                f"Number of receiver processes is not equal to the number of receiver keys"
            )

    splitted_rank = get_splitted_rank(all_participants_type, rank, type)
    # Create numpy arrays for sending and receiving
    send_array = np.array([splitted_rank], dtype=np.int32)
    recv_array = np.empty(size, dtype=np.int32)
    comm.Allgather([send_array, MPI.INT], [recv_array, MPI.INT])
    all_participants_splitted_rank = recv_array.tolist()
    if rank == 0:
        print(all_participants_splitted_rank)
    
    # Create numpy arrays for sending and receiving keys
    send_key = np.array([keys[splitted_rank]], dtype=np.float32)
    recv_keys = np.empty(size, dtype=np.float32)
    comm.Allgather([send_key, MPI.FLOAT], [recv_keys, MPI.FLOAT])
    all_participants_keys = recv_keys.tolist()
    if rank == 0:
        print(all_participants_keys)
    
    validate_lambda_values(all_participants_keys, all_participants_type)
    partner_rank = find_keys_partner(all_participants_keys, rank)
    
    # Create numpy arrays for sending and receiving partner ranks
    send_partner = np.array([partner_rank], dtype=np.int32)
    recv_partners = np.empty(size, dtype=np.int32)
    comm.Allgather([send_partner, MPI.INT], [recv_partners, MPI.INT])
    all_participants_partners_rank = recv_partners.tolist()
    if rank == 0:
        print(all_participants_partners_rank)
    print(f"Partner rank for rank {rank} is {partner_rank}")
    return partner_rank
