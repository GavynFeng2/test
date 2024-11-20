from cangjie_dataset_b3 import ETL952Train, ETL952Test, ETL952Val
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    images, class_labels, string_labels = zip(*batch)
    
    images = torch.stack(images)
    class_labels = torch.tensor(class_labels)
    
    # Convert string labels to numerical labels (e.g., ASCII values or a predefined mapping)
    string_labels_numerical = [[ord(char) - ord('a') + 1 for char in label] for label in string_labels]
    string_labels_tensors = [torch.tensor(label) for label in string_labels_numerical]
    string_labels_padded = pad_sequence(string_labels_tensors, batch_first=True, padding_value=0)
    
    target_lengths = torch.tensor([len(label) for label in string_labels_tensors])
    
    return images, class_labels, string_labels_padded, target_lengths

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
    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, collate_fn=collate_fn)
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
    test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, collate_fn=collate_fn)
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
    val_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, collate_fn=collate_fn)
    return val_data_loader

