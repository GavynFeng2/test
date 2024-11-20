from cangjie_dataset import ETL952Train, ETL952Test, ETL952Val, ETL952Labels
from torch.utils.data import DataLoader

def get_training_loader(batch_size=128, num_workers=4, shuffle=True):
    """
    return training data loader
    
    Args:
        batch_size:
        num_workers:
        shuffle:

    Returns: train_data_loader: torch dataloader object
    """

    # train_set = ETL952Train(root_dir="pytorch-cifar100")
    train_set = ETL952Train(root_dir="")
    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
    return train_data_loader

def get_test_loader(batch_size=128, num_workers=4, shuffle=True):
    """
    return test data loader
    
    Args:
        batch_size:
        num_workers:
        shuffle:

    Returns: test_data_loader: torch dataloader object
    """

    # test_set = ETL952Test(root_dir="pytorch-cifar100")
    test_set = ETL952Test(root_dir="")
    test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
    return test_data_loader

def get_val_loader(batch_size=128, num_workers=4, shuffle=True):
    """
    return val data loader
    
    Args:
        batch_size:
        num_workers:
        shuffle:

    Returns: val_data_loader: torch dataloader object
    """

    # test_set = ETL952Test(root_dir="pytorch-cifar100")
    val_set = ETL952Val(root_dir="")
    val_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
    return val_data_loader

