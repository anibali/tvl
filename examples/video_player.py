import importlib
import logging
import sys
import tkinter as tk
import traceback
from functools import lru_cache
from pathlib import Path
from queue import Queue
from threading import Thread
from time import time, sleep
from tkinter import filedialog
from tkinter import ttk

import PIL.Image
import PIL.ImageTk
import torch

import tvl

data_dir = Path(__file__).parent.parent.joinpath('data')


class VideoThread(Thread):
    def __init__(self, log):
        super().__init__(name='VideoThread')
        self.log = log
        self.vl = None
        self.replace_vl = None
        self.running = True
        self.paused = True
        self.time_multiplier = 1
        self.seeking = False
        self.frame_index = 0
        self.cur_image = None
        self.queue = Queue(maxsize=1)

    def set_video_loader(self, vl):
        self.replace_vl = vl

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

    def run(self):
        last_time = time()
        acc_time = 0
        while self.running:
            if self.replace_vl is not None:
                self.vl = self.replace_vl
                self.replace_vl = None
                self.seeking = True
                self.log.debug(f'Switched backend to {self.vl.backend.__class__.__name__}')
            if self.vl is None:
                sleep(0.01)
                continue
            this_time = time()
            frame_time = (1.0 / self.vl.frame_rate) * self.time_multiplier
            if self.seeking:
                self.seeking = False
                self.log.debug(f'Seeking to frame {self.frame_index}')
                self.vl.seek_to_frame(self.frame_index)
                try:
                    self._read_frame()
                except EOFError:
                    self.log.debug(f'Hit end of file, seeking to frame 0')
                    self.vl.seek(0)
                    self.frame_index = 0
                    self._read_frame()
            elif not self.paused:
                acc_time += this_time - last_time
                while acc_time > frame_time:
                    try:
                        self.frame_index += 1
                        self._read_frame()
                    except EOFError:
                        self.log.debug(f'Hit end of file, seeking to frame 0')
                        self.vl.seek(0)
                        self.frame_index = 0
                        self._read_frame()
                    acc_time -= frame_time
            sleep(max(frame_time - (time() - this_time), 0.005))
            last_time = this_time


class MainApp(tk.Tk):
    def __init__(self, video_thread):
        super().__init__()

        self.device = torch.device('cuda')
        torch.empty(0).to(self.device)  # Ensure device is initialised

        self.file_path = None
        self.backend_factory = None
        self.backend_opts = {}

        self.video_thread = video_thread

        self.wm_title('Video player')
        self.geometry('1280x720')

        # Don't show hidden files in the file dialog by default.
        try:
            # call a dummy dialog with an impossible option to initialize the file
            # dialog without really getting a dialog window; this will throw a
            # TclError, so we need a try...except :
            try:
                self.tk.call('tk_getOpenFile', '-thisoptiondoesnotexist')
            except tk.TclError:
                pass
            # now set the magic variables accordingly
            self.tk.call('set', '::tk::dialog::file::showHiddenBtn', '1')
            self.tk.call('set', '::tk::dialog::file::showHiddenVar', '0')
        except:
            pass

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
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img = img.copy()
            img.thumbnail((canvas_width, canvas_height))
            photo_image = PIL.ImageTk.PhotoImage(img)
            self.canvas.create_image((canvas_width / 2, canvas_height / 2), anchor=tk.CENTER,
                                     image=photo_image)
            self.canvas.image = photo_image
            self.canvas.update()

    def update_video_loader(self, file_path=None, device=None, backend_factory=None, backend_opts=None):
        if backend_factory is not None:
            self.backend_factory = backend_factory
        if device is not None:
            self.device = device
        if file_path is not None:
            self.file_path = file_path
        if backend_opts is not None:
            self.backend_opts = backend_opts

        if not self.file_path:
            return

        try:
            if self.backend_factory is not None:
                tvl.set_backend_factory(self.device.type, self.backend_factory)
            vl = tvl.VideoLoader(self.file_path, self.device, backend_opts=self.backend_opts)
            self.video_thread.set_video_loader(vl)
            self.spn_frame_index.configure(to=vl.n_frames - 1)
            self.title(self.file_path.name)
        except Exception:
            print(f'Failed to load video: {self.file_path}')
            traceback.print_exc()

    def do_seek(self):
        try:
            self.video_thread.seek_to_frame(self.var_frame_index.get())
        except tk.TclError:
            pass

    def do_select_video_file(self):
        start_dir = None
        if self.file_path is not None:
            start_dir = str(self.file_path.parent)
        elif Path(data_dir).is_dir():
            start_dir = str(data_dir)
        filename = filedialog.askopenfilename(initialdir=start_dir, title='Select a video file')
        if filename:
            self.update_video_loader(Path(filename), self.device)

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

        backend_items = {}
        for device_type, backend_names in tvl._known_backends.items():
            for backend_name in backend_names:
                try:
                    module_name, class_name = backend_name.rsplit('.', 1)
                    module = importlib.import_module(module_name)
                    backend_items[f'{device_type} {class_name}'] = (torch.device(device_type), getattr(module, class_name)())
                except ImportError:
                    pass

        cmb_backends = ttk.Combobox(toolbar, values=list(backend_items.keys()))
        cmb_backends.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        def on_backend_change(event):
            device, backend_factory = backend_items[cmb_backends.get()]
            self.update_video_loader(device=device, backend_factory=backend_factory)
        cmb_backends.bind("<<ComboboxSelected>>", on_backend_change)

        cmb_backends.current(0)
        on_backend_change(None)

        self.var_720p = tk.IntVar()
        def on_toggle_720p(*args):
            if self.var_720p.get():
                backend_opts = dict(out_width=1280, out_height=720)
            else:
                backend_opts = {}
            self.update_video_loader(backend_opts=backend_opts)
        self.var_720p.trace_variable('w', on_toggle_720p)
        self.var_720p.set(1)
        chk_force_720p = tk.Checkbutton(toolbar, text="Force 720p", variable=self.var_720p)
        chk_force_720p.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        self.var_slomo = tk.IntVar()
        def on_toggle_slomo(*args):
            if self.var_slomo.get():
                self.video_thread.time_multiplier = 10
            else:
                self.video_thread.time_multiplier = 1
        self.var_slomo.trace_variable('w', on_toggle_slomo)
        self.var_slomo.set(0)
        chk_slomo = tk.Checkbutton(toolbar, text="10x slow motion", variable=self.var_slomo)
        chk_slomo.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        return toolbar


def main():
    log = logging.Logger('video_player')
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

    video_thread = VideoThread(log)
    video_thread.start()
    app = MainApp(video_thread)
    app.mainloop()
    video_thread.stop()
    video_thread.join()


if __name__ == '__main__':
    main()
