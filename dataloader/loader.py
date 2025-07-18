import numpy as np
from jittor.dataset import DataLoader


class Loader:
    def __init__(self, dataset, mode, batch_size, num_workers, drop_last, sampler, data_index=None):

        self.loader = DataLoader(dataset, batch_size=batch_size,
                                                    num_workers=num_workers, 
                                                    drop_last=drop_last, collate_batch=collate_events)
        # if mode == "training":
        #     self.loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
        #                                               num_workers=num_workers, 
        #                                               drop_last=drop_last, collate_fn=collate_events)
        # else:
        #     self.loader = DataLoader(dataset, batch_size=batch_size,
        #                                               num_workers=num_workers, 
        #                                               drop_last=drop_last, collate_fn=collate_events_test)
    def __iter__(self):
        for data in self.loader:
            yield data

    def __len__(self):
        return len(self.loader)

def collate_events(data):
    labels = []
    pos_events = []
    neg_events = []
    idx_batch = 0
    
    for d in data:
        for idx in range(len(d[0])):
            label = d[0][idx]
            if label.ndim == 1:
                label = label[np.newaxis, :]
            lb = np.concatenate([label, idx_batch * np.ones((len(label), 1), dtype=np.float32)], 1)
            labels.append(lb)
            idx_batch += 1
        
        pos_events.append(d[1])  
        neg_events.append(d[2])
    
    labels = np.concatenate(labels, 0)
    
    return labels, pos_events, neg_events


