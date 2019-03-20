from concurrent.futures import Executor

from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import Sampler, BatchSampler, RandomSampler, SequentialSampler


class BatchDataIter:
    def __init__(self, loader):
        self.loader = loader
        self.batch_iter = iter(loader.batch_sampler)
        self.batch_buffer = []
        self.max_buffer_len = 2

    def _prepare_future_batches(self, batch_iter):
        while len(self.batch_buffer) < self.max_buffer_len:
            try:
                batch_indices = next(batch_iter)
            except StopIteration:
                break
            future_batch = [self.loader.dataset[index] for index in batch_indices]
            self.batch_buffer.append(future_batch)

    def __next__(self):
        self._prepare_future_batches(self.batch_iter)
        if len(self.batch_buffer) == 0:
            raise StopIteration()
        future_batch = self.batch_buffer.pop(0)
        batch = [example.result() for example in future_batch]
        self._prepare_future_batches(self.batch_iter)
        return self.loader.collate(batch)

    def __len__(self):
        return len(self.loader)


class BatchDataLoader:
    def __init__(self, dataset, *, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 collate_fn=default_collate, drop_last=False):
        """Loads batches of data from an asynchronous dataset.

        Args:
            dataset (AsyncDataset):
            batch_size (int):
            shuffle (bool):
            sampler (Sampler):
            batch_sampler (Sampler):
            collate_fn (function):
            drop_last (bool):
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate = collate_fn

        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with shuffle')

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive with batch_size, '
                                 'shuffle, sampler, and drop_last')
        else:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __iter__(self):
        return BatchDataIter(self)

    def __len__(self):
        return len(self.batch_sampler)


class AsyncDataset:
    def __init__(self, dataset, executor):
        """Wraps a synchronous dataset to enable asynchronous access to examples.

        Args:
            dataset (Dataset): the synchronous dataset to wrap
            executor (Executor): the executor supplying a pool of workers
        """
        self.dataset = dataset
        self.executor = executor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.executor.submit(self.dataset.__getitem__, index)
