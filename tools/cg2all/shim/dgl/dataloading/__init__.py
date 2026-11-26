class GraphDataLoader:  # unused; scripts iterate datasets directly
    def __init__(self, dataset, **kw):
        self.dataset = dataset

    def __iter__(self):
        for i in range(len(self.dataset)):
            yield self.dataset[i]

    def __len__(self):
        return len(self.dataset)
