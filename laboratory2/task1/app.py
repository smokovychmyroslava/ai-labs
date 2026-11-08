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
    RadioGroup,
    Spinner,
    TableColumn,
)
from bokeh.plotting import figure

state = {
    "pending_gray": None,   
    "samples": [],          
    "w": None,              
    "b": None,              
    "cursor": 0,            
    "epoch": 0,            
    "epoch_errors": 0,     
    "history": [],          
    "pending": None,      
    "unknown_gray": None,
}

CLASS_NAME = {1: "A", -1: "B"}

rows_spinner = Spinner(title="Рядків сітки", low=1, high=50, step=1, value=4, width=140)
cols_spinner = Spinner(title="Стовпців сітки", low=1, high=50, step=1, value=5, width=140)
thresh_spinner = Spinner(title="Поріг бінаризації (0-255)", low=0, high=255, step=1,
                         value=128, width=180)
mode_radio = RadioGroup(
    labels=["Подавати на вхід нормований вектор X_норм",
            "Подавати на вхід абсолютний вектор X"],
    active=0,
)
grid_settings = row(rows_spinner, cols_spinner, thresh_spinner)

wrap_formatter = HTMLTemplateFormatter(
    template='<div style="white-space:normal;word-break:break-word;line-height:1.4;padding:4px 0;"><%= value %></div>'
)


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


def features(gray):
    mask = binarize(gray, int(thresh_spinner.value))
    vec, rb, cb = compute_vector(mask, int(rows_spinner.value), int(cols_spinner.value))
    vn = normalize(vec)
    x = np.array(vn if mode_radio.active == 0 else vec, dtype=float)
    return vec, vn, x, rb, cb


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


def make_small_figure(title="", size=90):
    f = figure(title=title, width=size, height=size + 25, toolbar_location=None, tools="")
    f.axis.visible = False
    f.grid.visible = False
    return f


def predict(x):
    s = float(np.dot(state["w"], x) - state["b"])
    return s, (1 if s > 0 else -1)


def adjust(x, y, d):
    r = float(lr_spinner.value)
    state["w"] = state["w"] + r * (d - y) * x


def grid_html(values, digits=3):
    n = len(values)
    rows, cols = int(rows_spinner.value), int(cols_spinner.value)
    if rows * cols != n:
        rows, cols = 1, n
    trs = []
    for r in range(rows):
        tds = []
        for c in range(cols):
            v = values[r * cols + c]
            color = "#c6e6c6" if v >= 0 else "#f4c7c3"
            tds.append(
                f"<td style='background:{color};padding:2px 6px;text-align:right'>"
                f"{v:.{digits}f}</td>"
            )
        trs.append("<tr>" + "".join(tds) + "</tr>")
    return ("<table style='border-collapse:collapse;font-size:12px' border='1'>"
            + "".join(trs) + "</table>")


def show_weights():
    w, b = state["w"], state["b"]
    if w is None:
        weights_div.text = "<i>Ваги ще не ініціалізовано.</i>"
        return
    weights_div.text = (
        f"<b>Поточний стан перцептрона:</b> поріг b = {b:.4f}<br>"
        f"Ваги w (по клітинках сітки; зелений ≥ 0, червоний &lt; 0):<br>{grid_html(w)}"
    )


def weights_before_after(w0, b0, w1, b1, title):
    return (
        f"<b>{title}</b>"
        "<table><tr><td style='vertical-align:top;padding-right:20px'>"
        f"До: b = {b0:.4f}<br>{grid_html(w0)}</td>"
        "<td style='vertical-align:top'>"
        f"Після: b = {b1:.4f}<br>{grid_html(w1)}</td></tr></table>"
    )


def check_ready(div):
    if state["w"] is None:
        div.text = "<span style='color:red'>Спочатку ініціалізуйте ваги перцептрона.</span>"
        return False
    dim = int(rows_spinner.value) * int(cols_spinner.value)
    if len(state["w"]) != dim:
        div.text = (
            "<span style='color:red'>Розмірність сітки змінилась після ініціалізації ваг "
            f"({len(state['w'])} ≠ {dim}). Ініціалізуйте ваги заново.</span>"
        )
        return False
    return True


lr_spinner = Spinner(title="Швидкість навчання r (< 1)", low=0.001, high=0.999,
                     step=0.05, value=0.5, width=200)
max_epochs_spinner = Spinner(title="Макс. кількість епох", low=1, high=100000, step=100,
                             value=1000, width=180)
init_button = Button(label="Ініціалізувати ваги (0 < w < 1, 0 < b < 1)",
                     button_type="primary", width=320)
weights_div = Div(text="<i>Ваги ще не ініціалізовано.</i>", width=900)


def reset_training(clear_weights=False):
    state.update(cursor=0, epoch=0, epoch_errors=0, history=[], pending=None)
    step_div.text = ""
    update_div.text = ""
    all_div.text = ""
    all_source.data = dict(n=[], d=[], y=[], s=[], ok=[])
    if clear_weights:
        state["w"] = None
        state["b"] = None
    show_weights()
    redraw_history()


def init_weights():
    dim = int(rows_spinner.value) * int(cols_spinner.value)
    state["w"] = np.random.uniform(0.0, 1.0, dim)
    state["b"] = float(np.random.uniform(0.0, 1.0))
    reset_training()


init_button.on_click(init_weights)

params_section = column(
    Div(text="<h3>2. Параметри перцептрона</h3>"),
    row(lr_spinner, max_epochs_spinner),
    init_button,
    weights_div,
)

class_radio = RadioGroup(labels=["Клас A (d = +1)", "Клас B (d = −1)"], active=0)
add_sample_button = Button(label="Додати зразок", button_type="primary", width=160)
clear_samples_button = Button(label="Очистити зразки", button_type="warning", width=160)
train_status = Div(
    text="<i>Завантажте зображення зразка, оберіть клас і натисніть «Додати зразок».</i>",
    width=900)
train_file_container = row()
gallery_container = column()

samples_source = ColumnDataSource(data=dict(n=[], cls=[], x=[], xnorm=[]))
samples_table = DataTable(
    source=samples_source,
    columns=[
        TableColumn(field="n", title="№", width=40),
        TableColumn(field="cls", title="Клас", width=60),
        TableColumn(field="x", title="X (абсолютний вектор)", width=380, formatter=wrap_formatter),
        TableColumn(field="xnorm", title="X_норм (нормований вектор)", width=380,
                    formatter=wrap_formatter),
    ],
    width=900, height=200, row_height=70, autosize_mode="none",
)


def on_train_file_upload(attr, old, new):
    try:
        gray = decode_image(new)
    except Exception as exc:  # noqa: BLE001
        train_status.text = f"<span style='color:red'>Помилка завантаження: {exc}</span>"
        return
    state["pending_gray"] = gray
    train_status.text = (f"Зображення завантажено ({gray.shape[1]}×{gray.shape[0]} px). "
                         f"Оберіть клас і натисніть «Додати зразок».")


def make_train_file_input():
    fi = FileInput(accept=".bmp,.png,.jpg,.jpeg", title="Зображення зразка")
    fi.on_change("value", on_train_file_upload)
    return fi


def reset_train_file_input():
    train_file_container.children = [make_train_file_input()]


def redraw_samples():
    thumbs = []
    for i, s in enumerate(state["samples"]):
        f = make_small_figure(f"#{i + 1}: {CLASS_NAME[s['d']]}")
        draw_image_with_grid(f, s["gray"])
        thumbs.append(f)
    gallery_container.children = [row(*thumbs[i:i + 8]) for i in range(0, len(thumbs), 8)]

    ns, cls, xs, xns = [], [], [], []
    for i, s in enumerate(state["samples"]):
        vec, vn, _, _, _ = features(s["gray"])
        ns.append(i + 1)
        cls.append(CLASS_NAME[s["d"]])
        xs.append(str(vec))
        xns.append(str(vn))
    samples_source.data = dict(n=ns, cls=cls, x=xs, xnorm=xns)


def add_sample():
    gray = state["pending_gray"]
    if gray is None:
        train_status.text = "<span style='color:red'>Спочатку завантажте зображення зразка.</span>"
        return
    d = 1 if class_radio.active == 0 else -1
    state["samples"].append({"gray": gray, "d": d})
    state["pending_gray"] = None
    reset_train_file_input()
    redraw_samples()
    reset_training()  
    na = sum(1 for s in state["samples"] if s["d"] == 1)
    nb = len(state["samples"]) - na
    train_status.text = (f"Додано зразок класу {CLASS_NAME[d]}. "
                         f"Усього зразків: {len(state['samples'])} (A: {na}, B: {nb}).")


def clear_samples():
    state["samples"] = []
    state["pending_gray"] = None
    reset_train_file_input()
    redraw_samples()
    reset_training()
    train_status.text = "Усі зразки видалено."


reset_train_file_input()
add_sample_button.on_click(add_sample)
clear_samples_button.on_click(clear_samples)

samples_section = column(
    Div(text="<h3>1. Навчальні зразки двох класів</h3>"),
    row(train_file_container, class_radio, column(add_sample_button, clear_samples_button)),
    train_status,
    gallery_container,
    samples_table,
)

classify_next_button = Button(label="Класифікувати наступний зразок", button_type="primary", width=240)
confirm_button = Button(label="Вчитель: підтвердити", button_type="success", width=180)
reject_button = Button(label="Вчитель: спростувати (корекція ваг)", button_type="danger", width=260)
step_div = Div(text="", width=900)
update_div = Div(text="", width=900)

autotrain_button = Button(label="Автонавчання за мітками класів (до збіжності)",
                          button_type="success", width=340)
classify_all_button = Button(label="Класифікувати всю вибірку (вихідний вектор Y)",
                             button_type="default", width=340)
conv_div = Div(text="", width=900)
all_div = Div(text="", width=900)

all_source = ColumnDataSource(data=dict(n=[], d=[], y=[], s=[], ok=[]))
all_table = DataTable(
    source=all_source,
    columns=[
        TableColumn(field="n", title="№", width=40),
        TableColumn(field="d", title="Бажаний d", width=100),
        TableColumn(field="y", title="Вихід y", width=100),
        TableColumn(field="s", title="S = Σwᵢxᵢ − b", width=160),
        TableColumn(field="ok", title="Результат", width=120),
    ],
    width=540, height=180, autosize_mode="none", index_position=None,
)

hist_source = ColumnDataSource(data=dict(epoch=[], errors=[]))
history_fig = figure(title="Збіжність: кількість помилок за епохами", width=600, height=260,
                     toolbar_location=None, tools="")
history_fig.xaxis.axis_label = "Епоха"
history_fig.yaxis.axis_label = "Помилок"
history_fig.line("epoch", "errors", source=hist_source, line_width=2)
history_fig.scatter("epoch", "errors", source=hist_source, size=6)


def redraw_history():
    hist = state["history"]
    hist_source.data = dict(epoch=[h[0] for h in hist], errors=[h[1] for h in hist])
    if not hist:
        conv_div.text = "<i>Завершених епох ще немає (епоха — повний прохід по всіх зразках).</i>"
    elif hist[-1][1] == 0:
        conv_div.text = (f"<b style='color:green'>Навчання збіглося:</b> епоха {hist[-1][0]} "
                         f"пройшла без жодної помилки. Усього епох: {len(hist)}.")
    else:
        conv_div.text = (f"<b style='color:#b36b00'>Збіжності ще немає:</b> в останній епосі "
                         f"({hist[-1][0]}) було помилок: {hist[-1][1]}. Усього епох: {len(hist)}.")


def record_step(err):
    state["epoch_errors"] += err
    state["cursor"] += 1
    if state["cursor"] >= len(state["samples"]):
        state["epoch"] += 1
        state["history"].append((state["epoch"], state["epoch_errors"]))
        state["cursor"] = 0
        state["epoch_errors"] = 0
    redraw_history()


def classify_next():
    if not state["samples"]:
        step_div.text = "<span style='color:red'>Немає навчальних зразків.</span>"
        return
    if not check_ready(step_div):
        return
    idx = state["cursor"]
    s = state["samples"][idx]
    vec, vn, x, _, _ = features(s["gray"])
    sum_, y = predict(x)
    state["pending"] = {"idx": idx, "x": x, "y": y}
    update_div.text = ""
    step_div.text = (
        f"<b>Зразок #{idx + 1}</b> (мітка в навчальній послідовності: {CLASS_NAME[s['d']]})<br>"
        f"X = ({'; '.join(map(str, vec))})<br>"
        f"X_норм = ({'; '.join(map(str, vn))})<br>"
        f"S = Σ wᵢ·xᵢ − b = {sum_:.5f} &nbsp;→&nbsp; y = {y:+d} "
        f"(перцептрон відносить зразок до класу <b>{CLASS_NAME[y]}</b>)<br>"
        f"<i>Підтвердіть результат (якщо він правильний) або спростуйте його (якщо хибний).</i>"
    )


def teacher_confirm():
    p = state["pending"]
    if p is None:
        step_div.text = "<span style='color:red'>Спочатку натисніть «Класифікувати наступний зразок».</span>"
        return
    label = state["samples"][p["idx"]]["d"]
    if p["y"] != label:
        update_div.text = (
            f"<span style='color:red'>Підтвердити не можна: перцептрон дав клас {CLASS_NAME[p['y']]}, "
            f"а зразок належить класу {CLASS_NAME[label]}. Натисніть «спростувати».</span>")
        return
    update_div.text = "Результат підтверджено, ваги не змінюються (d = y)."
    state["pending"] = None
    record_step(0)


def teacher_reject():
    p = state["pending"]
    if p is None:
        step_div.text = "<span style='color:red'>Спочатку натисніть «Класифікувати наступний зразок».</span>"
        return
    label = state["samples"][p["idx"]]["d"]
    if p["y"] == label:
        update_div.text = (
            f"<span style='color:red'>Спростувати не можна: клас {CLASS_NAME[p['y']]} визначено "
            f"правильно. Натисніть «підтвердити».</span>")
        return
    w0, b0 = state["w"].copy(), state["b"]
    adjust(p["x"], p["y"], label)
    update_div.text = weights_before_after(
        w0, b0, state["w"], state["b"],
        f"Результат спростовано: d = {label:+d}, корекція ваг Δw = r·(d − y)·x")
    show_weights()
    state["pending"] = None
    record_step(1)


def classify_all():
    samples = state["samples"]
    if not samples:
        all_div.text = "<span style='color:red'>Немає навчальних зразків.</span>"
        return
    if not check_ready(all_div):
        return
    ns, ds, ys, ss, oks = [], [], [], [], []
    errors = 0
    for i, smp in enumerate(samples):
        _, _, x, _, _ = features(smp["gray"])
        s, y = predict(x)
        bad = y != smp["d"]
        errors += bad
        ns.append(i + 1)
        ds.append(f"{smp['d']:+d} ({CLASS_NAME[smp['d']]})")
        ys.append(f"{y:+d} ({CLASS_NAME[y]})")
        ss.append(f"{s:.5f}")
        oks.append("✗ помилка" if bad else "✓")
    all_source.data = dict(n=ns, d=ds, y=ys, s=ss, ok=oks)
    all_div.text = (
        f"<b>Бажаний вектор D</b> = ({'; '.join(d.split()[0] for d in ds)})<br>"
        f"<b>Вихідний вектор Y</b> = ({'; '.join(y.split()[0] for y in ys)})<br>"
        f"Помилок: <b>{errors}</b> з {len(samples)}."
    )


def autotrain():
    samples = state["samples"]
    if len(samples) < 2 or len({s["d"] for s in samples}) < 2:
        conv_div.text = "<span style='color:red'>Потрібні зразки обох класів (A і B).</span>"
        return
    if not check_ready(conv_div):
        return
    xs = [features(s["gray"])[2] for s in samples]
    w0, b0 = state["w"].copy(), state["b"]
    state.update(cursor=0, epoch_errors=0, pending=None)
    converged = False
    for _ in range(int(max_epochs_spinner.value)):
        errors = 0
        for s, x in zip(samples, xs):
            _, y = predict(x)
            if y != s["d"]:
                adjust(x, y, s["d"])
                errors += 1
        state["epoch"] += 1
        state["history"].append((state["epoch"], errors))
        if errors == 0:
            converged = True
            break
    update_div.text = weights_before_after(w0, b0, state["w"], state["b"],
                                           "Зміна ваг за час автонавчання")
    show_weights()
    redraw_history()
    if not converged:
        conv_div.text += ("<br><span style='color:#b36b00'>Ліміт епох вичерпано. Спробуйте змінити r, "
                          "збільшити ліміт епох або перевірити, "
                          "чи класи лінійно роздільні.</span>")


classify_next_button.on_click(classify_next)
confirm_button.on_click(teacher_confirm)
reject_button.on_click(teacher_reject)
autotrain_button.on_click(autotrain)
classify_all_button.on_click(classify_all)

training_section = column(
    Div(text="<h3>3. Навчання з учителем</h3>"),
    row(classify_next_button, confirm_button, reject_button),
    step_div,
    update_div,
    row(autotrain_button, classify_all_button),
    conv_div,
    all_div,
    all_table,
    history_fig,
)

unknown_file_input = FileInput(accept=".bmp,.png,.jpg,.jpeg", title="Невідоме зображення")
classify_unknown_button = Button(label="Класифікувати невідомий образ", button_type="success", width=260)
unknown_status = Div(text="<i>Завантажте невідоме зображення.</i>", width=900)
unknown_result = Div(text="", width=900)
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
    state["unknown_gray"] = gray
    unknown_status.text = f"Завантажено ({gray.shape[1]}×{gray.shape[0]} px)."
    unknown_result.text = ""
    draw_image_with_grid(unknown_fig, gray)


def classify_unknown():
    gray = state["unknown_gray"]
    if gray is None:
        unknown_status.text = "<span style='color:red'>Спочатку завантажте невідоме зображення.</span>"
        return
    if not check_ready(unknown_status):
        return
    vec, vn, x, rb, cb = features(gray)
    draw_image_with_grid(unknown_fig, gray, rb, cb)
    s, y = predict(x)
    unknown_result.text = (
        f"<b>X</b> = ({'; '.join(map(str, vec))})<br>"
        f"<b>X_норм</b> = ({'; '.join(map(str, vn))})<br>"
        f"S = Σ wᵢ·xᵢ − b = {s:.5f}, &nbsp; y = {y:+d}<br>"
        f"<b>Результат: клас {CLASS_NAME[y]}</b> (за поточним станом перцептрона)"
    )


unknown_file_input.on_change("value", on_unknown_file_upload)
classify_unknown_button.on_click(classify_unknown)

unknown_section = column(
    Div(text="<h3>4. Класифікація невідомого образу</h3>"),
    row(unknown_file_input, classify_unknown_button),
    unknown_status,
    row(unknown_fig, unknown_result),
)


def on_params_change(attr, old, new):
    redraw_samples()
    reset_training(clear_weights=True)
    unknown_result.text = ""
    if state["samples"] or state["w"] is not None:
        train_status.text = ("<span style='color:#b36b00'>Параметри ознак змінено: вектори перераховано, "
                             "ваги та історію навчання скинуто. Ініціалізуйте ваги заново.</span>")


for _w in (rows_spinner, cols_spinner, thresh_spinner):
    _w.on_change("value", on_params_change)
mode_radio.on_change("active", on_params_change)

layout = column(
    Div(text="<h2>Лабораторна робота №2, завдання 1. Перцептрон Розенблата (два класи)</h2>"),
    Div(text="<b>Параметри побудови вектора ознак:</b>"),
    grid_settings,
    mode_radio,
    samples_section,
    params_section,
    training_section,
    unknown_section,
)

redraw_history()
curdoc().add_root(layout)
curdoc().title = "Лабораторна робота №2, завдання 1 — Перцептрон Розенблата (Bokeh)"