import tkinter as tk
from functools import lru_cache
from pathlib import Path
from queue import Queue
from threading import Thread
from time import time, sleep
from tkinter import filedialog

import PIL.Image
import PIL.ImageTk
import torch

from tvl import VideoLoader

data_dir = Path(__file__).parent.parent.joinpath('tests/data')


class VideoThread(Thread):
    def __init__(self):
        super().__init__(name='VideoThread')
        self.vl = None
        self.running = True
        self.paused = True
        self.seeking = False
        self.frame_index = 0
        self.cur_image = None
        self.queue = Queue(maxsize=1)

    def set_video_loader(self, vl):
        self.vl = vl
        self.frame_index = 0
        self._read_frame()

    def seek_to_frame(self, frame_index):
        self.frame_index = frame_index
        self.seeking = True

    def stop(self):
        self.running = False

    def _read_frame(self):
        self.cur_image = self.vl.read_frame()
        while not self.queue.empty():
            self.queue.get()
        self.queue.put((self.cur_image, self.frame_index))
        self.frame_index += 1

    def run(self):
        last_time = time()
        acc_time = 0
        while self.running:
            if self.vl is None:
                sleep(10)
                continue
            this_time = time()
            frame_time = 1.0 / self.vl.frame_rate
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

        self.device = torch.device('cuda:0')
        torch.empty(0).to(self.device)  # Ensure device is initialised

        self.video_thread = video_thread

        self.wm_title('Video player')
        self.geometry('1280x720')

        self.var_frame_index = tk.IntVar(value=0)

        root_frame = tk.Frame(self)
        root_frame.pack(fill=tk.BOTH, expand=True)

        global_toolbar = self._make_global_toolbar(root_frame)
        global_toolbar.pack(side=tk.TOP, fill=tk.X)

        self.canvas = tk.Canvas(root_frame, background='#111111')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.first_file_open = True

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

    def do_seek(self):
        try:
            self.video_thread.seek_to_frame(self.var_frame_index.get())
        except tk.TclError:
            pass

    def do_select_video_file(self):
        start_dir = None
        if self.first_file_open and Path(data_dir).is_dir():
            start_dir = str(data_dir)
        filename = filedialog.askopenfilename(initialdir=start_dir, title='Select a video file')
        if filename:
            self.video_thread.set_video_loader(VideoLoader(filename, self.device))
            self.spn_frame_index.configure(to=self.video_thread.vl.n_frames - 1)
            self.first_file_open = False

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

        btn_open = tk.Button(toolbar, text='Open...', command=self.do_select_video_file)
        btn_open.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        add_label('Frame:')
        n_frames = self.video_thread.vl.n_frames - 1 if self.video_thread.vl is not None else 0
        spn_frame_index = tk.Spinbox(
            toolbar, textvariable=self.var_frame_index,
            wrap=True, from_=0, to=n_frames,
            command=self.do_seek)
        def on_key_spinbox(event):
            if event.keysym == 'Return':
                self.do_seek()
        spn_frame_index.bind('<Key>', on_key_spinbox)
        spn_frame_index.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
        self.spn_frame_index = spn_frame_index

        btn_pause = tk.Button(toolbar, text='Seek', command=self.do_seek)
        btn_pause.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        def on_click_pause():
            self.video_thread.paused = not self.video_thread.paused

        btn_pause = tk.Button(toolbar, text='Play/Pause', command=on_click_pause)
        btn_pause.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        return toolbar


def main():
    video_thread = VideoThread()
    video_thread.start()
    app = MainApp(video_thread)
    app.mainloop()
    video_thread.stop()
    video_thread.join()


if __name__ == '__main__':
    main()
