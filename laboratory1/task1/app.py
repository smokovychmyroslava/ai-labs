import base64
import io
import os
from datetime import datetime

import numpy as np
from PIL import Image

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    ColumnDataSource,
    DataTable,
    Div,
    FileInput,
    Spinner,
    TableColumn,
)
from bokeh.plotting import figure

state = {
    "gray": None,
    "mask": None,
    "filename": None,
    "vector": None,
    "vector_norm": None,
}

file_input = FileInput(accept=".bmp,.png,.jpg,.jpeg", title="Завантажити зображення")
rows_spinner = Spinner(title="Рядків сітки", low=1, high=50, step=1, value=4, width=140)
cols_spinner = Spinner(title="Стовпців сітки", low=1, high=50, step=1, value=5, width=140)
thresh_spinner = Spinner(title="Поріг бінаризації (0-255)", low=0, high=255, step=1,
                         value=128, width=180)

build_button = Button(label="Побудувати вектор ознак", button_type="primary", width=230)

status_div = Div(text="<i>Завантажте зображення, щоб почати.</i>", width=780)
vector_div = Div(text="", width=780)

image_fig = figure(title="Зображення із сіткою поділу", width=380, height=380,
                    toolbar_location=None, tools="")
image_fig.axis.visible = False
image_fig.grid.visible = False

bar_fig = figure(title="Абсолютний вектор ознак X", width=380, height=250,
                 toolbar_location=None, tools="", x_range=["0"])
bar_fig.xaxis.major_label_orientation = 1.0

norm_fig = figure(title="Нормований вектор ознак Xнорм", width=380, height=250,
                   toolbar_location=None, tools="", x_range=["0"], y_range=(0, 1))
norm_fig.xaxis.major_label_orientation = 1.0

table_source = ColumnDataSource(data=dict(cell=[], count=[], norm=[]))
columns = [
    TableColumn(field="cell", title="Область"),
    TableColumn(field="count", title="К-сть пікселів"),
    TableColumn(field="norm", title="Нормоване значення"),
]
data_table = DataTable(source=table_source, columns=columns, width=380, height=250)


def decode_image(b64_value: str) -> np.ndarray:
    if b64_value.strip().startswith("data:") and "," in b64_value:
        b64_value = b64_value.split(",", 1)[1]
    raw = base64.b64decode(b64_value)
    img = Image.open(io.BytesIO(raw)).convert("L")
    return np.array(img, dtype=np.uint8)


def binarize(gray: np.ndarray, threshold: int) -> np.ndarray:
    return gray < threshold


def compute_vector(mask: np.ndarray, rows: int, cols: int):
    h, w = mask.shape
    row_bounds = np.linspace(0, h, rows + 1).astype(int)
    col_bounds = np.linspace(0, w, cols + 1).astype(int)
    vector = []
    for r in range(rows):
        for c in range(cols):
            cell = mask[row_bounds[r]:row_bounds[r + 1], col_bounds[c]:col_bounds[c + 1]]
            vector.append(int(cell.sum()))
    return vector, row_bounds, col_bounds


def normalize(vector):
    total = sum(vector)
    if total == 0:
        return [0.0 for _ in vector]
    return [round(v / total, 4) for v in vector]


def gray_to_rgba(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    rgba = np.empty((h, w), dtype=np.uint32)
    view = rgba.view(dtype=np.uint8).reshape((h, w, 4))
    view[:, :, 0] = gray
    view[:, :, 1] = gray
    view[:, :, 2] = gray
    view[:, :, 3] = 255
    return np.flipud(rgba)


def redraw_image(row_bounds=None, col_bounds=None):
    image_fig.renderers = []
    gray = state["gray"]
    if gray is None:
        return
    h, w = gray.shape
    image_fig.x_range.start, image_fig.x_range.end = 0, w
    image_fig.y_range.start, image_fig.y_range.end = 0, h
    image_fig.image_rgba(image=[gray_to_rgba(gray)], x=0, y=0, dw=w, dh=h)

    if row_bounds is not None:
        for rb in row_bounds:
            y = h - rb
            image_fig.line(x=[0, w], y=[y, y], line_color="red", line_width=1)
    if col_bounds is not None:
        for cb in col_bounds:
            image_fig.line(x=[cb, cb], y=[0, h], line_color="red", line_width=1)


def redraw_vectors(vector, vector_norm):
    cats = [str(i + 1) for i in range(len(vector))]

    bar_fig.renderers = []
    bar_fig.x_range.factors = cats
    bar_fig.vbar(x=cats, top=vector, width=0.8, fill_color="#2b8cbe")

    norm_fig.renderers = []
    norm_fig.x_range.factors = cats
    norm_fig.vbar(x=cats, top=vector_norm, width=0.8, fill_color="#31a354")

    table_source.data = dict(cell=cats, count=vector, norm=vector_norm)

    vector_div.text = (
        f"<b>X</b> = ({'; '.join(map(str, vector))})<br>"
        f"<b>X_норм</b> = ({'; '.join(map(str, vector_norm))})"
    )


def on_filename_change(attr, old, new):
    state["filename"] = new


def on_file_upload(attr, old, new):
    try:
        gray = decode_image(new)
    except Exception as exc:  # noqa: BLE001
        status_div.text = f"<span style='color:red'>Помилка завантаження: {exc}</span>"
        return
    state["gray"] = gray
    name = state["filename"] or "image"
    status_div.text = (
        f"Завантажено: <b>{name}</b> "
        f"({gray.shape[1]}×{gray.shape[0]} px)"
    )
    redraw_image()
    build_vector()


def build_vector():
    if state["gray"] is None:
        status_div.text = "<span style='color:red'>Спочатку завантажте зображення.</span>"
        return
    mask = binarize(state["gray"], thresh_spinner.value)
    state["mask"] = mask

    rows, cols = rows_spinner.value, cols_spinner.value
    vector, row_bounds, col_bounds = compute_vector(mask, rows, cols)
    vector_norm = normalize(vector)

    state["vector"], state["vector_norm"] = vector, vector_norm

    redraw_image(row_bounds, col_bounds)
    redraw_vectors(vector, vector_norm)


file_input.on_change("filename", on_filename_change)
file_input.on_change("value", on_file_upload)
build_button.on_click(build_vector)

controls = column(
    file_input,
    row(rows_spinner, cols_spinner, thresh_spinner),
    build_button,
    status_div,
)

layout = column(
    controls,
    row(image_fig, column(bar_fig, norm_fig)),
    row(data_table, vector_div),
)

curdoc().add_root(layout)
curdoc().title = "Завдання 1 — Вектор ознак зображення (Bokeh)"