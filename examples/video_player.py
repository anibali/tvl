import tkinter as tk
from pathlib import Path

import PIL.Image
import PIL.ImageTk
import torch
from functools import lru_cache
from threading import Thread
from time import time, sleep
from queue import Queue

from tvl import VideoLoader

data_dir = Path(__file__).parent.parent.joinpath('tests/data')
video_filename = str(data_dir.joinpath('board_game-h264.mkv'))


class VideoThread(Thread):
    def __init__(self, vl):
        super().__init__(name='VideoThread')
        self.vl = vl
        self.running = True
        self.paused = False
        self.seeking = False
        self.frame_index = 0
        self.cur_image = None
        self.queue = Queue(maxsize=30)

    def seek_to_frame(self, frame_index):
        self.frame_index = frame_index
        self.seeking = True

    def stop(self):
        self.running = False

    def _read_frame(self):
        self.cur_image = self.vl.read_frame()
        self.queue.put((self.cur_image, self.frame_index))
        self.frame_index += 1

    def run(self):
        last_time = time()
        acc_time = 0
        frame_time = 1.0 / self.vl.frame_rate
        while self.running:
            this_time = time()
            if self.seeking:
                self.seeking = False
                self.vl.seek_to_frame(self.frame_index)
                try:
                    self._read_frame()
                except EOFError:
                    self.vl.seek(0)
                    self.frame_index = 0
                    self._read_frame()
            elif not self.paused:
                acc_time += this_time - last_time
                while acc_time > frame_time:
                    try:
                        self._read_frame()
                    except EOFError:
                        self.vl.seek(0)
                        self.frame_index = 0
                        self._read_frame()
                    acc_time -= frame_time
            sleep(max(frame_time - (time() - this_time), 0))
            last_time = this_time


class MainApp(tk.Tk):
    def __init__(self, video_thread):
        super().__init__()

        self.video_thread = video_thread

        self.wm_title('Video player')
        self.geometry('1280x720')

        self.vl = VideoLoader(video_filename, torch.device('cuda:0'))

        self.var_frame_index = tk.IntVar(value=0)

        root_frame = tk.Frame(self)
        root_frame.pack(fill=tk.BOTH, expand=True)

        global_toolbar = self._make_global_toolbar(root_frame)
        global_toolbar.pack(side=tk.TOP, fill=tk.X)

        self.canvas = tk.Canvas(root_frame, background='#111111')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.do_update()

    @lru_cache(maxsize=1)
    def show_frame(self, rgb_tensor):
        if rgb_tensor is None:
            return
        rgb_bytes = (rgb_tensor * 255).round_().byte().cpu()
        img = PIL.Image.fromarray(rgb_bytes.permute(1, 2, 0).numpy(), 'RGB')
        if img is not None:
            img = img.copy()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img.thumbnail((canvas_width, canvas_height))
            photo_image = PIL.ImageTk.PhotoImage(img)
            self.canvas.create_image((canvas_width / 2, canvas_height / 2), anchor=tk.CENTER,
                                     image=photo_image)
            self.canvas.image = photo_image
            self.canvas.update()

    def do_update(self):
        rgb_tensor = None
        frame_index = 0
        while not self.video_thread.queue.empty():
            rgb_tensor, frame_index = self.video_thread.queue.get()
        if rgb_tensor is not None:
            self.show_frame(rgb_tensor)
            self.var_frame_index.set(frame_index)
        self.after(10, self.do_update)

    def _make_global_toolbar(self, master):
        toolbar = tk.Frame(master, bd=1, relief=tk.RAISED)

        def add_label(text):
            opts = dict(text=text) if isinstance(text, str) else dict(textvariable=text)
            label = tk.Label(toolbar, **opts)
            label.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
            return label

        def do_seek_to_frame():
            try:
                self.video_thread.seek_to_frame(self.var_frame_index.get())
            except tk.TclError:
                pass

        add_label('Frame:')
        spn_frame_index = tk.Spinbox(
            toolbar, textvariable=self.var_frame_index,
            wrap=True, from_=0, to=self.vl.n_frames - 1,
            command=do_seek_to_frame)
        def on_key_spinbox(event):
            if event.keysym == 'Return':
                do_seek_to_frame()
        spn_frame_index.bind('<Key>', on_key_spinbox)
        spn_frame_index.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        btn_pause = tk.Button(toolbar, text='Seek', command=do_seek_to_frame)
        btn_pause.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        def on_click_pause():
            self.video_thread.paused = not self.video_thread.paused

        btn_pause = tk.Button(toolbar, text='Play/Pause', command=on_click_pause)
        btn_pause.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        return toolbar


def main():
    device = torch.device('cuda:0')
    torch.empty(0).to(device)  # Ensure device is initialised
    vl = VideoLoader(video_filename, device)
    video_thread = VideoThread(vl)
    video_thread.start()
    app = MainApp(video_thread)
    app.mainloop()
    video_thread.stop()
    video_thread.join()


if __name__ == '__main__':
    main()
