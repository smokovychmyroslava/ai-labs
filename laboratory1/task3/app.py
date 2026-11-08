import base64
import html
import io

import numpy as np
from PIL import Image

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    Div,
    FileInput,
    RadioGroup,
    Spinner,
    TextInput,
)
from bokeh.plotting import figure


state = {
    "train_gray_pending": None,   
    "classes": {},               
    "unknown": {"gray": None, "vector": None, "vector_norm": None},
}

METHOD_CENTERS = 0  
METHOD_BOXES = 1    


rows_spinner = Spinner(title="Рядків сітки", low=1, high=50, step=1, value=4, width=140)
cols_spinner = Spinner(title="Стовпців сітки", low=1, high=50, step=1, value=5, width=140)
thresh_spinner = Spinner(title="Поріг бінаризації (0-255)", low=0, high=255, step=1,
                         value=128, width=180)

method_radio = RadioGroup(
    labels=[
        "Варіант 1: геометричні центри кластерів (метод порівняння з еталоном, Евклідова норма)",
        "Варіант 2: мінімум/максимум компонент (метод ортогональних областей)",
    ],
    active=METHOD_CENTERS,
)

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


def make_small_figure(size=90):
    f = figure(width=size, height=size, toolbar_location=None, tools="")
    f.axis.visible = False
    f.grid.visible = False
    return f

def compute_cluster_stats(vectors_norm):
    n = len(vectors_norm)
    dim = len(vectors_norm[0])
    center = [sum(v[i] for v in vectors_norm) / n for i in range(dim)]
    std = [
        (sum((v[i] - center[i]) ** 2 for v in vectors_norm) / n) ** 0.5
        for i in range(dim)
    ]
    vmin = [min(v[i] for v in vectors_norm) for i in range(dim)]
    vmax = [max(v[i] for v in vectors_norm) for i in range(dim)]
    return {
        "n": n,
        "center": [round(x, 4) for x in center],
        "std": [round(x, 4) for x in std],
        "min": [round(x, 4) for x in vmin],
        "max": [round(x, 4) for x in vmax],
    }


def euclidean(a, b):
    return float(np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))


def box_violation(vec, vmin, vmax):
    total, inside = 0.0, True
    for x, lo, hi in zip(vec, vmin, vmax):
        if x < lo:
            total += lo - x
            inside = False
        elif x > hi:
            total += x - hi
            inside = False
    return inside, total


class_name_input = TextInput(title="Назва класу (напр. '1', '2', '3')", width=200)
add_sample_button = Button(label="Додати зразок до навчальної послідовності",
                            button_type="primary", width=260)
train_status = Div(text="<i>Завантажте зображення зразка та вкажіть клас.</i>", width=900)

train_file_container = row()  
classes_container = column() 


def make_train_file_input():
    fi = FileInput(accept=".bmp,.png,.jpg,.jpeg", title="Зображення зразка")
    fi.on_change("value", on_train_file_upload)
    return fi


def reset_train_file_input():
    train_file_container.children = [make_train_file_input()]


def on_train_file_upload(attr, old, new):
    try:
        gray = decode_image(new)
    except Exception as exc:  # noqa: BLE001
        train_status.text = f"<span style='color:red'>Помилка завантаження: {exc}</span>"
        return
    state["train_gray_pending"] = gray
    train_status.text = (
        f"Зображення зразка завантажено ({gray.shape[1]}×{gray.shape[0]} px). "
        f"Вкажіть клас і натисніть «Додати зразок»."
    )


def redraw_classes():
    blocks = []
    for name, data in state["classes"].items():
        samples = data["samples"]
        stats = data["stats"]

        thumbs = []
        for s in samples:
            f = make_small_figure(90)
            draw_image_with_grid(f, s["gray"])
            thumbs.append(f)
        thumbs_row = row(*thumbs) if thumbs else Div(text="<i>(немає зразків)</i>")

        stats_html = (
            f"<b>Клас «{html.escape(name)}»</b> — зразків: {stats['n']}<br>"
            f"Геометричний центр (середнє X_норм): ({'; '.join(map(str, stats['center']))})<br>"
            f"Середньоквадратичне відхилення: ({'; '.join(map(str, stats['std']))})<br>"
            f"Мінімум по компонентах: ({'; '.join(map(str, stats['min']))})<br>"
            f"Максимум по компонентах: ({'; '.join(map(str, stats['max']))})"
        )
        blocks.append(column(thumbs_row, Div(text=stats_html, width=900)))
    classes_container.children = blocks


def add_sample():
    gray = state["train_gray_pending"]
    if gray is None:
        train_status.text = "<span style='color:red'>Спочатку завантажте зображення зразка.</span>"
        return
    name = class_name_input.value.strip()
    if not name:
        train_status.text = "<span style='color:red'>Введіть назву класу.</span>"
        return

    mask = binarize(gray, thresh_spinner.value)
    vector, _, _ = compute_vector(mask, rows_spinner.value, cols_spinner.value)
    vector_norm = normalize(vector)

    cls = state["classes"].setdefault(name, {"samples": [], "stats": None})
    cls["samples"].append({"gray": gray, "vector": vector, "vector_norm": vector_norm})
    cls["stats"] = compute_cluster_stats([s["vector_norm"] for s in cls["samples"]])

    state["train_gray_pending"] = None
    reset_train_file_input()
    redraw_classes()

    train_status.text = (
        f"Додано зразок до класу «{html.escape(name)}» "
        f"(усього зразків у класі: {len(cls['samples'])}). "
        f"X = ({'; '.join(map(str, vector))}), "
        f"X_норм = ({'; '.join(map(str, vector_norm))})."
    )


reset_train_file_input()
add_sample_button.on_click(add_sample)

training_section = column(
    Div(text="<h3>1-2. Навчальна послідовність і поточна статистика кластерів</h3>"),
    row(class_name_input, train_file_container, add_sample_button),
    train_status,
    classes_container,
)

unknown_file_input = FileInput(accept=".bmp,.png,.jpg,.jpeg", title="Невідоме зображення")
build_unknown_button = Button(label="Побудувати вектор ознак", button_type="primary", width=220)
unknown_status = Div(text="<i>Завантажте невідоме зображення.</i>", width=900)
unknown_vector_div = Div(text="", width=900)

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
    Div(text="<h3>3. Невідомий образ</h3>"),
    row(unknown_file_input, build_unknown_button),
    unknown_status,
    row(unknown_fig, unknown_vector_div),
)

classify_button = Button(label="Класифікувати", button_type="success", width=220)
classification_div = Div(text="", width=900)


def classify():
    if not state["classes"]:
        classification_div.text = "<span style='color:red'>Немає жодного класу з навчальної послідовності.</span>"
        return
    if state["unknown"]["vector_norm"] is None:
        classification_div.text = (
            "<span style='color:red'>Спочатку побудуйте вектор невідомого образу.</span>"
        )
        return

    u = state["unknown"]["vector_norm"]
    method = method_radio.active

    rows_html = []
    if method == METHOD_CENTERS:
        best_name, best_d = None, float("inf")
        for name, data in state["classes"].items():
            d = euclidean(u, data["stats"]["center"])
            rows_html.append(f"<tr><td>{html.escape(name)}</td><td>{round(d, 4)}</td></tr>")
            if d < best_d:
                best_name, best_d = name, d
        table = (
            "<table border='1' cellpadding='6' cellspacing='0'>"
            "<tr><th>Клас</th><th>Евклідова відстань до центру</th></tr>"
            + "".join(rows_html) + "</table>"
        )
        verdict = (
            f"<b>Метод:</b> геометричні центри кластерів (Евклідова норма)<br>"
            f"<b>Результат класифікації:</b> клас «{html.escape(str(best_name))}» "
            f"(найменша відстань d = {round(best_d, 4)})"
        )
    else:
        matches = []
        viol_by_name = {}
        for name, data in state["classes"].items():
            inside, viol = box_violation(u, data["stats"]["min"], data["stats"]["max"])
            viol_by_name[name] = viol
            rows_html.append(
                f"<tr><td>{html.escape(name)}</td>"
                f"<td>{'так' if inside else 'ні'}</td>"
                f"<td>{round(viol, 4)}</td></tr>"
            )
            if inside:
                matches.append(name)
        table = (
            "<table border='1' cellpadding='6' cellspacing='0'>"
            "<tr><th>Клас</th><th>Належить області?</th><th>Величина порушення меж</th></tr>"
            + "".join(rows_html) + "</table>"
        )
        if len(matches) == 1:
            best_name = matches[0]
            note = "однозначно потрапляє в ортогональну область цього класу"
        elif len(matches) > 1:
            best_name = min(matches, key=lambda n: viol_by_name[n])
            note = (
                f"одночасно потрапляє в області {len(matches)} класів — "
                f"обрано клас із найменшим відхиленням"
            )
        else:
            best_name = min(viol_by_name, key=lambda n: viol_by_name[n])
            note = "не потрапляє в жодну область — обрано клас із найменшим порушенням меж"
        verdict = (
            f"<b>Метод:</b> ортогональні області (мінімум/максимум компонент)<br>"
            f"<b>Результат класифікації:</b> клас «{html.escape(str(best_name))}» ({note})"
        )

    classification_div.text = verdict + "<br><br>" + table


classify_button.on_click(classify)

classification_section = column(
    Div(text="<h3>4. Класифікація невідомого образу</h3>"),
    classify_button,
    classification_div,
)

layout = column(
    Div(text="<h2>Завдання 3. Система розпізнавання з навчанням (кластеризація)</h2>"),
    Div(text="<b>Спільні параметри сітки поділу зображення:</b>"),
    grid_settings,
    Div(text="<b>Метод кластеризації / класифікації:</b>"),
    method_radio,
    training_section,
    unknown_section,
    classification_section,
)

curdoc().add_root(layout)
curdoc().title = "Завдання 3 — Навчання і класифікація (Bokeh)"