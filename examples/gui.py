import tkinter as tk
from pathlib import Path

import PIL.Image
import PIL.ImageTk
import torch
from functools import lru_cache

from tvl import VideoLoader

data_dir = Path(__file__).parent.parent.joinpath('tests/data')
video_filename = str(data_dir.joinpath('board_game-h264.mkv'))


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.wm_title('Video GUI')
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
    def load_frame(self, frame_index):
        try:
            rgb = self.vl.pick_frames([frame_index])[0]
            rgb_bytes = (rgb * 255).round_().byte().cpu()
            img = PIL.Image.fromarray(rgb_bytes.permute(1, 2, 0).numpy(), 'RGB')
            return img
        except Exception:
            import traceback
            traceback.print_exc()
            return None

    def do_update(self):
        img = self.load_frame(self.var_frame_index.get())
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

        self.after(100, self.do_update)


    def _make_global_toolbar(self, master):
        toolbar = tk.Frame(master, bd=1, relief=tk.RAISED)

        def add_label(text):
            opts = dict(text=text) if isinstance(text, str) else dict(textvariable=text)
            label = tk.Label(toolbar, **opts)
            label.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
            return label

        add_label('Frame:')
        spn_frame_index = tk.Spinbox(
            toolbar, textvariable=self.var_frame_index,
            wrap=True, from_=0, to=self.vl.n_frames - 1)
        def validate_spinbox(new_value):
            return new_value.isdigit()
        vcmd = (self.register(validate_spinbox), '%P')
        spn_frame_index.configure(validate='key', validatecommand=vcmd)
        spn_frame_index.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        return toolbar


def main():
    app = MainApp()
    app.mainloop()


if __name__ == '__main__':
    main()
