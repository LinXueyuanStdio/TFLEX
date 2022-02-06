import torch


def speed_up():
    torch.cuda.emptyCache()

def how_to_speed_up():
    info = """
    1. for your dataloader: pin_memory == True, num_worker >= 8
    2. choose faster optimizer: AdamW
    """
    print(info)
