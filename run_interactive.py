from pathlib import Path
import sys
import os
import logging
import traceback
from contextlib import contextmanager
import requests
from io import BytesIO

from PIL import Image
from rich.console import Console
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Vertical, Container
from textual.widgets import Input, Button, Log, Label

import gemnr.gemnr as gemnr_api


class GemNREditingApp(App):

    CSS = """
    Screen {
        overflow: auto;
    }
    #main {
        layout: grid;
        grid-size: 6;
        grid-rows: 5 5 5 5 5 5 1fr;
    }
    #row {
        column-span: 2;
    }
    #seed-full, #resolution-full, #anchor_idx-full {
        column-span: 2;
    }
    #token-full, #output_folder-full {
        column-span: 3;
    }
    #edit_text_prompt-full, #inputs-full, #anchor_input_ref-full, #output {
        column-span: 6;
    }
    #quit {
        column-span: 5;
    }
    """

    default_input = "./assets/stone_horse"
    default_edit_text = "Change the horse statue to a lion statue"
    default_anchor_idx = 1
    default_seed = 0
    default_output = "./results"
    default_resolution = 512
    default_token = os.environ.get("HF_TOKEN", "")
    gemnr_initialized = False
    gem_nr = None

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("enter", "run", "Run"),
        ("tab", "focus_next", "Next"),
        ("shift+tab", "focus_previous", "Previous"),
        ("down", "focus_next", "Next"),
        ("up", "focus_previous", "Previous"),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="main"):
            yield labeled_input(
                value=self.default_output,
                label="Output folder",
                id="output_folder",
            )
            yield labeled_input(
                value=self.default_token,
                label="Hugging Face token",
                id="token",
            )
            yield labeled_input(
                value=self.default_anchor_idx,
                label="Anchor img idx",
                id="anchor_idx",
            )
            yield labeled_input(
                value=self.default_resolution,
                label="Resolution",
                id="resolution",
            )
            yield labeled_input(
                value=self.default_seed,
                label="Seed",
                id="seed",
            )
            yield labeled_input(
                value=self.default_input,
                label="Input folder or list of image paths",
                id="inputs",
            )
            yield labeled_input(
                value=self.default_edit_text,
                label="Edit text prompt",
                id="edit_text_prompt",
            )
            yield labeled_input(
                label="(optional) Path to a reference image for an anchor edit",
                id="anchor_input_ref",
            )
            yield Button("Edit images", id="run", variant="primary")
            yield Button("Quit", id="quit", variant="warning")
            yield Log(id="output", auto_scroll=True)

    def on_mount(self):
        self.curr_worker = None
        input_widget = self.query_one("#inputs")
        input_widget.focus()

    @on(Button.Pressed, "#run")
    async def handle_run(self) -> None:
        output_folder = self.query_one("#output_folder", Input).value
        token = self.query_one("#token", Input).value
        anchor_idx = self.query_one("#anchor_idx", Input).value
        resolution = self.query_one("#resolution", Input).value
        seed = self.query_one("#seed", Input).value
        inputs = self.query_one("#inputs", Input).value
        anchor_input_ref = self.query_one("#anchor_input_ref", Input).value
        edit_text_prompt = self.query_one("#edit_text_prompt", Input).value
        log = self.query_one("#output", Log)
        log.clear()
        with capture_all(self, self.query_one(Log)) as console:
            try:
                self._check_inputs(
                    inputs,
                    edit_text_prompt,
                    anchor_idx,
                    output_folder,
                    resolution,
                    seed,
                    token,
                )

                self.query_one("#run", Button).disabled = True

                resolution = int(resolution)
                seed = int(seed)
                anchor_idx = int(anchor_idx)
                self.curr_worker = self._gemnr_setup(resolution=resolution, seed=seed, token=token)
                await self.curr_worker.wait()
                
                im_pil_list, im_names = self._gemnr_prepare_inputs(inputs=inputs)
                anchor_cond_pil = None if not anchor_input_ref else self._gemnr_prepare_inputs(inputs=anchor_input_ref)[0][0]
                
                self.curr_worker = self._gemnr_run_once(
                    im_pil_list=im_pil_list,
                    edit_text_prompt=edit_text_prompt,
                    anchor_idx=anchor_idx,
                    anchor_cond_pil=anchor_cond_pil,
                    output_folder=output_folder,
                )
                out_im_pil_list = await self.curr_worker.wait()
                self.curr_worker = None
                self.query_one("#run", Button).disabled = False
                
                self._gemnr_save_results(
                    im_pil_list=im_pil_list,
                    out_im_pil_list=out_im_pil_list,
                    im_names=im_names,
                    edit_text_prompt=edit_text_prompt,
                    output_folder=output_folder,
                    seed=seed,
                )

            except Exception as e:
                traceback.print_exc()
                self.query_one("#run", Button).disabled = False
                return


    @on(Button.Pressed, "#quit")
    def handle_quit(self) -> None:
        if self.curr_worker:
            self.curr_worker.cancel()
        self.exit()
    
    def _check_inputs(self, *inputs_str) -> bool:
        if any(map(lambda s: s == "", inputs_str)):
            raise ValueError("Fill-in the inputs!")

    @work(thread=True)
    def _gemnr_setup(self, resolution: int, seed: int, token: str):
        if not self.gemnr_initialized:
            print("Initializing GeM-NR...")
            self.gem_nr = gemnr_api.GemNR(
                resolution=resolution, seed=seed, token=token, lazy_init=False
            )
            self.gemnr_initialized = True
        elif resolution != self.gem_nr.resolution:
            print("Re-initializing GeM-NR with new resolution...")
            self.gem_nr = gemnr_api.GemNR(
                resolution=resolution, seed=seed, token=token
            )
        elif seed != self.gem_nr.seed:
            print("Re-seeding GeM-NR...")
            self.gem_nr.set_seed(seed)

    @work(thread=True)
    def _gemnr_run_once(
        self,
        im_pil_list: list[Image.Image],
        edit_text_prompt: str,
        anchor_idx: int,
        anchor_cond_pil: Image.Image | None,
        output_folder: str,
    ):
        print("Editing...")
        out_im_pil_list = self.gem_nr.edit(
            im_pil_list,
            edit_text_prompt=edit_text_prompt,
            anchor_idx=anchor_idx,
            anchor_cond_pil=anchor_cond_pil
        )
        return out_im_pil_list

    def _gemnr_prepare_inputs(self, inputs: str):
        im_pil_list, im_names = input_to_imgs(inputs)
        im_pil_list = [
            self.gem_nr.crop_resize(im_pil) for im_pil in im_pil_list
        ]
        return im_pil_list, im_names
    
    def _gemnr_save_results(self,
        im_pil_list: list[Image.Image],
        out_im_pil_list: list[Image.Image],
        im_names: list[str],
        edit_text_prompt: str,
        output_folder: str,
        seed: int,
    ):
        edit_id = "_".join(
            edit_text_prompt.lower()
            .replace(",", "")
            .replace(".", "")
            .split(" ")[:20]
        )
        log_dir = Path(f"{output_folder}/{edit_id}")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        for in_im, out_im, im_name in zip(
            im_pil_list, out_im_pil_list, im_names
        ):
            out_im.save(log_dir / f"edited{seed:02d}_{im_name}.jpg")
            in_im.save(log_dir / f"unedited_{im_name}.jpg")
        print(f"Saved to: {output_folder}/{edit_id}")


def labeled_input(
    label: str,
    id: str,
    value: any = "",
    width: int = "100%",
) -> Vertical:
    label_widget = Label(f"{label}:", shrink=False)
    input_widget = Input(value=str(value), id=id)
    widget = Vertical(label_widget, input_widget, id=f"{id}-full")
    widget.styles.width = width
    return widget


class LogSink:
    def __init__(self, app, log):
        self.app = app
        self.log = log

    def write(self, text):
        if text:
            try:
                self.app.call_from_thread(self.log.write, text)
            except Exception:
                self.log.write(text)

    def flush(self):
        pass


class LogHandler(logging.Handler):
    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def emit(self, record):
        msg = self.format(record)
        self.sink.write(msg)


@contextmanager
def capture_all(app, log):
    sink = LogSink(app, log)

    # stdout/stderr
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = sink

    # logging
    handler = LogHandler(sink)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    root = logging.getLogger()
    old_handlers = root.handlers[:]
    root.handlers = [handler]

    # rich console
    console = Console(file=sink, force_terminal=True, color_system="truecolor")

    try:
        yield console
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        root.handlers = old_handlers


def input_to_imgs(inputs: str) -> list[Image.Image]:
    img_suffixes = [".jpg", ".jpeg", ".png", ".heic"]
    if " " in inputs or any(
        [inputs.lower().endswith(s) for s in img_suffixes]
    ):
        if inputs.startswith("[") and inputs.endswith("]"):
            inputs = inputs[1:-1]
        im_paths = inputs.split(",")
    else:
        im_paths = [
            str(p)
            for p in Path(inputs).glob("*")
            if p.suffix.lower() in img_suffixes
        ]
        im_paths = sorted(im_paths)

    assert len(im_paths) > 0, f"No valid image paths found in {inputs}"

    if any(p.lower().endswith(".heic") for p in im_paths):
        from pillow_heif import register_heif_opener

        register_heif_opener()

    im_pil_list = []
    for im_path in im_paths:
        if im_path.lower().startswith(("http://", "https://")):
            response = requests.get(im_path)
            response.raise_for_status()
            img_data = BytesIO(response.content)
        else:
            img_data = im_path
        im_pil_list.append(Image.open(img_data).convert("RGB"))

    im_names = [Path(im_path).stem for im_path in im_paths]
    return im_pil_list, im_names


if __name__ == "__main__":
    GemNREditingApp().run(inline=True)
