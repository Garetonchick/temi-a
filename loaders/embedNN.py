import loaders

def get_dataset(dataset, datapath='./data', arch="PaSST", train=True):
    return getattr(loaders, dataset)(datapath=datapath, arch=arch)