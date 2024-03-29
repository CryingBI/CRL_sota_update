import torch
from torch.utils.data import Dataset, DataLoader

class hidden_set(Dataset):
    def __init__(self, data, hidden, config=None):
        self.data = data

        self.hidden = hidden
        
        self.config = config
        self.bert = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], idx)

    def collate_fn(self, data):

        label = torch.tensor([item[0]['relation'] for item in data])
        tokens = [torch.tensor(item[0]['tokens']) for item in data]
        ind = torch.tensor([item[1] for item in data])
        self.hidden = torch.cat(self.hidden)
        return (
            label,
            tokens,
            self.hidden,
            ind
        )
    
def get_hidden_loader(config, data, hidden, shuffle = False, drop_last = False, batch_size = None):

    dataset = hidden_set(data, hidden, config)

    if batch_size == None:
        batch_size = min(config.batch_size, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)

    return data_loader