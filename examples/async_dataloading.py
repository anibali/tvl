from concurrent.futures import ThreadPoolExecutor

from time import perf_counter
from pathlib import Path
import torch
from torch.utils.data import Dataset

from tvl import VideoLoader
from tvl.async import AsyncDataset, BatchDataLoader


data_dir = Path(__file__).parent.parent.joinpath('data')
video_filename = str(data_dir.joinpath('board_game-h264.mkv'))


class VideoDataset(Dataset):
    def __init__(self, clips, device=torch.device('cuda:0')):
        self.clips = clips
        self.device = device

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        video_filename, frame_indices = self.clips[index]
        vl = VideoLoader(video_filename, self.device)
        frames = torch.stack(vl.select_frames_as_list(frame_indices), 0)
        return dict(
            frames=frames,
            example_index=index,
        )


def main():
    device = torch.device('cuda:0')
    torch.empty(0).to(device)  # Initialise CUDA manually so that it doesn't interfere with timing.

    dataset = VideoDataset([(video_filename, list(range(40)))] * 8, device=device)
    # NOTE: The VideoLoader object can deadlock if more than one thread try to use it at once
    async_dataset = AsyncDataset(dataset, ThreadPoolExecutor(max_workers=1))
    loader = BatchDataLoader(async_dataset, batch_size=2, shuffle=False)

    start_time = perf_counter()
    for batch in loader:
        pass
    end_time = perf_counter()
    print(f'Time taken: {end_time - start_time} seconds')


if __name__ == '__main__':
    main()
