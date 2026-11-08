import base64
import io

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
    HTMLTemplateFormatter,
    Spinner,
    TableColumn,
    TextInput,
)
from bokeh.plotting import figure

state = {
    "etalon_gray_pending": None,
    "etalons": [],  
    "unknown": {"gray": None, "vector": None, "vector_norm": None},
}

rows_spinner = Spinner(title="Рядків сітки", low=1, high=50, step=1, value=4, width=140)
cols_spinner = Spinner(title="Стовпців сітки", low=1, high=50, step=1, value=5, width=140)
thresh_spinner = Spinner(title="Поріг бінаризації (0-255)", low=0, high=255, step=1,
                         value=128, width=180)

grid_settings = row(rows_spinner, cols_spinner, thresh_spinner)


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


def draw_image_with_grid(fig, gray, row_bounds=None, col_bounds=None):
    fig.renderers = []
    if gray is None:
        return
    h, w = gray.shape
    fig.x_range.start, fig.x_range.end = 0, w
    fig.y_range.start, fig.y_range.end = 0, h
    fig.image_rgba(image=[gray_to_rgba(gray)], x=0, y=0, dw=w, dh=h)
    if row_bounds is not None:
        for rb in row_bounds:
            y = h - rb
            fig.line(x=[0, w], y=[y, y], line_color="red", line_width=1)
    if col_bounds is not None:
        for cb in col_bounds:
            fig.line(x=[cb, cb], y=[0, h], line_color="red", line_width=1)


def euclidean(a, b):
    return float(np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))


def manhattan(a, b):
    return float(sum(abs(x - y) for x, y in zip(a, b)))


def chebyshev(a, b):
    return float(max(abs(x - y) for x, y in zip(a, b)))

class_name_input = TextInput(title="Назва класу (напр. '0', '1', 'А')", width=200)
add_etalon_button = Button(label="Додати еталон", button_type="primary", width=200)
clear_etalons_button = Button(label="Очистити всі еталони", button_type="warning", width=200)
etalon_controls = row() 

etalon_status = Div(text="<i>Завантажте зображення еталону та вкажіть назву класу.</i>",
                    width=780)

etalons_gallery = row()  

wrap_formatter = HTMLTemplateFormatter(
    template='<div style="white-space:normal;word-break:break-word;line-height:1.4;padding:4px 0;"><%= value %></div>'
)

etalons_source = ColumnDataSource(data=dict(name=[], x=[], xnorm=[]))
etalons_table = DataTable(
    source=etalons_source,
    columns=[
        TableColumn(field="name", title="Клас", width=60),
        TableColumn(field="x", title="X (абсолютний вектор)", width=360, formatter=wrap_formatter),
        TableColumn(field="xnorm", title="X_норм (нормований вектор)", width=360, formatter=wrap_formatter),
    ],
    width=780,
    height=150,
    row_height=90,
    autosize_mode="none",
)


def redraw_etalons_gallery():
    figs = []
    for et in state["etalons"]:
        f = figure(title=et["name"], width=150, height=150, toolbar_location=None, tools="")
        f.axis.visible = False
        f.grid.visible = False
        draw_image_with_grid(f, et["gray"])
        figs.append(f)
    etalons_gallery.children = figs


def update_etalons_table():
    etalons_source.data = dict(
        name=[e["name"] for e in state["etalons"]],
        x=[str(e["vector"]) for e in state["etalons"]],
        xnorm=[str(e["vector_norm"]) for e in state["etalons"]],
    )


def on_etalon_file_upload(attr, old, new):
    try:
        gray = decode_image(new)
    except Exception as exc:  
        etalon_status.text = f"<span style='color:red'>Помилка завантаження: {exc}</span>"
        return
    state["etalon_gray_pending"] = gray
    etalon_status.text = (
        f"Зображення еталону завантажено ({gray.shape[1]}×{gray.shape[0]} px). "
        f"Введіть назву класу і натисніть «Додати еталон»."
    )


def make_etalon_file_input():
    fi = FileInput(accept=".bmp,.png,.jpg,.jpeg", title="Зображення еталону")
    fi.on_change("value", on_etalon_file_upload)
    return fi


def reset_etalon_file_input():
    etalon_controls.children = [class_name_input, make_etalon_file_input(), add_etalon_button]


def add_etalon():
    gray = state["etalon_gray_pending"]
    if gray is None:
        etalon_status.text = "<span style='color:red'>Спочатку завантажте зображення еталону.</span>"
        return
    name = class_name_input.value.strip()
    if not name:
        etalon_status.text = "<span style='color:red'>Введіть назву класу.</span>"
        return

    mask = binarize(gray, thresh_spinner.value)
    vector, _, _ = compute_vector(mask, rows_spinner.value, cols_spinner.value)
    vector_norm = normalize(vector)

    state["etalons"].append(
        {"name": name, "gray": gray, "vector": vector, "vector_norm": vector_norm}
    )
    state["etalon_gray_pending"] = None
    class_name_input.value = ""
    reset_etalon_file_input()

    redraw_etalons_gallery()
    update_etalons_table()
    etalon_status.text = f"Додано еталон класу «{name}». Всього еталонів: {len(state['etalons'])}."


def clear_etalons():
    state["etalons"].clear()
    state["etalon_gray_pending"] = None
    redraw_etalons_gallery()
    update_etalons_table()
    results_source.data = dict(name=[], eucl=[], cheb=[], manh=[])
    classification_div.text = ""
    reset_etalon_file_input()
    etalon_status.text = "Усі еталони видалено."


reset_etalon_file_input()
add_etalon_button.on_click(add_etalon)
clear_etalons_button.on_click(clear_etalons)

etalons_section = column(
    Div(text="<h3>1. Еталонні образи класів</h3>"),
    etalon_controls,
    clear_etalons_button,
    etalon_status,
    etalons_gallery,
    etalons_table,
)

unknown_file_input = FileInput(accept=".bmp,.png,.jpg,.jpeg", title="Невідоме зображення")
build_unknown_button = Button(label="Побудувати вектор ознак", button_type="primary", width=220)
unknown_status = Div(text="<i>Завантажте невідоме зображення.</i>", width=780)
unknown_vector_div = Div(text="", width=780)

unknown_fig = figure(title="Невідомий образ із сіткою поділу", width=300, height=300,
                     toolbar_location=None, tools="")
unknown_fig.axis.visible = False
unknown_fig.grid.visible = False


def on_unknown_file_upload(attr, old, new):
    try:
        gray = decode_image(new)
    except Exception as exc:  
        unknown_status.text = f"<span style='color:red'>Помилка завантаження: {exc}</span>"
        return
    state["unknown"]["gray"] = gray
    state["unknown"]["vector"] = None
    state["unknown"]["vector_norm"] = None
    unknown_status.text = f"Завантажено ({gray.shape[1]}×{gray.shape[0]} px)."
    draw_image_with_grid(unknown_fig, gray)


def build_unknown_vector():
    gray = state["unknown"]["gray"]
    if gray is None:
        unknown_status.text = "<span style='color:red'>Спочатку завантажте невідоме зображення.</span>"
        return
    mask = binarize(gray, thresh_spinner.value)
    vector, row_bounds, col_bounds = compute_vector(mask, rows_spinner.value, cols_spinner.value)
    vector_norm = normalize(vector)

    state["unknown"]["vector"] = vector
    state["unknown"]["vector_norm"] = vector_norm

    draw_image_with_grid(unknown_fig, gray, row_bounds, col_bounds)
    unknown_vector_div.text = (
        f"<b>X</b> = ({'; '.join(map(str, vector))})<br>"
        f"<b>X_норм</b> = ({'; '.join(map(str, vector_norm))})"
    )


unknown_file_input.on_change("value", on_unknown_file_upload)
build_unknown_button.on_click(build_unknown_vector)

unknown_section = column(
    Div(text="<h3>2. Невідомий образ</h3>"),
    row(unknown_file_input, build_unknown_button),
    unknown_status,
    row(unknown_fig, unknown_vector_div),
)

classify_button = Button(label="Класифікувати", button_type="success", width=220)
classification_div = Div(text="", width=780)

results_source = ColumnDataSource(data=dict(name=[], eucl=[], cheb=[], manh=[]))
results_table = DataTable(
    source=results_source,
    columns=[
        TableColumn(field="name", title="Клас (еталон)"),
        TableColumn(field="eucl", title="Евклідова відстань"),
        TableColumn(field="cheb", title="Чебишевська відстань"),
        TableColumn(field="manh", title="Манхетенська відстань"),
    ],
    width=780,
    height=150,
)


def classify():
    if not state["etalons"]:
        classification_div.text = "<span style='color:red'>Немає жодного доданого еталону.</span>"
        return
    if state["unknown"]["vector_norm"] is None:
        classification_div.text = (
            "<span style='color:red'>Спочатку побудуйте вектор невідомого образу.</span>"
        )
        return

    u = state["unknown"]["vector_norm"]
    names, eucl_vals, cheb_vals, manh_vals = [], [], [], []
    best_eucl = best_cheb = best_manh = (None, float("inf"))

    for et in state["etalons"]:
        v = et["vector_norm"]
        de, dc, dm = euclidean(u, v), chebyshev(u, v), manhattan(u, v)
        names.append(et["name"])
        eucl_vals.append(round(de, 4))
        cheb_vals.append(round(dc, 4))
        manh_vals.append(round(dm, 4))
        if de < best_eucl[1]:
            best_eucl = (et["name"], de)
        if dc < best_cheb[1]:
            best_cheb = (et["name"], dc)
        if dm < best_manh[1]:
            best_manh = (et["name"], dm)

    results_source.data = dict(name=names, eucl=eucl_vals, cheb=cheb_vals, manh=manh_vals)
    classification_div.text = (
        f"<b>Евклідова норма:</b> клас «{best_eucl[0]}» (d = {round(best_eucl[1], 4)})<br>"
        f"<b>Чебишевська норма:</b> клас «{best_cheb[0]}» (d = {round(best_cheb[1], 4)})<br>"
        f"<b>Манхетенська норма:</b> клас «{best_manh[0]}» (d = {round(best_manh[1], 4)})"
    )


classify_button.on_click(classify)

classification_section = column(
    Div(text="<h3>3. Класифікація невідомого образу</h3>"),
    classify_button,
    results_table,
    classification_div,
)

layout = column(
    Div(text="<h2>Завдання 2. Система розпізнавання за методом порівняння з еталоном</h2>"),
    Div(text="<b>Спільні параметри сітки поділу зображення:</b>"),
    grid_settings,
    etalons_section,
    unknown_section,
    classification_section,
)

curdoc().add_root(layout)
curdoc().title = "Завдання 2 — Класифікація образів (Bokeh)"