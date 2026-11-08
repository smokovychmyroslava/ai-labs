import base64
import html
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
    TextInput,
)
from bokeh.plotting import figure

state = {
    "etalon_gray_pending": None,
    "etalons": [],      
    "unknown": {"gray": None, "vector": None, "vector_norm": None, "bipolar": None},
    "net": None,        
    "hop": None,        
    "ham": None,        
}

rows_spinner = Spinner(title="Рядків сітки", low=1, high=50, step=1, value=4, width=140)
cols_spinner = Spinner(title="Стовпців сітки", low=1, high=50, step=1, value=5, width=140)
thresh_spinner = Spinner(title="Поріг бінаризації (0-255)", low=0, high=255, step=1,
                         value=128, width=180)
eps_spinner = Spinner(title="ε для Хемінга (0 < ε < 1/m)", low=0.001, high=1.0, step=0.01,
                      value=0.3, width=200)
iter_spinner = Spinner(title="Макс. ітерацій", low=1, high=1000, step=10, value=100, width=140)
ham_mode_radio = RadioGroup(
    labels=["Хемінг: біполярні вектори −1/+1",
            "Хемінг: нормовані вектори X_норм"],
    active=0,
)

grid_settings = row(rows_spinner, cols_spinner, thresh_spinner, eps_spinner, iter_spinner)

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


def to_bipolar(vector):
    arr = np.array(vector, dtype=float)
    return np.where(arr > arr.mean(), 1, -1).astype(int)


def unit(vec):
    arr = np.array(vec, dtype=float)
    nrm = np.linalg.norm(arr)
    return arr / nrm if nrm > 0 else arr


def features(gray):
    mask = binarize(gray, int(thresh_spinner.value))
    vector, rb, cb = compute_vector(mask, int(rows_spinner.value), int(cols_spinner.value))
    return vector, normalize(vector), to_bipolar(vector), rb, cb


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


def make_small_figure(title="", size=110):
    f = figure(title=title, width=size, height=size + 25, toolbar_location=None, tools="")
    f.axis.visible = False
    f.grid.visible = False
    return f


def bipolar_grid_html(v, cell=12):
    rows, cols = int(rows_spinner.value), int(cols_spinner.value)
    if rows * cols != len(v):
        rows, cols = 1, len(v)
    trs = "".join(
        "<tr>" + "".join(
            f"<td style='width:{cell}px;height:{cell}px;border:1px solid #bbb;"
            f"background:{'#222' if v[r * cols + c] > 0 else '#fff'}'></td>"
            for c in range(cols)) + "</tr>"
        for r in range(rows))
    return f"<table style='border-collapse:collapse'>{trs}</table>"


def bipolar_text(v):
    return "".join("■" if x > 0 else "□" for x in v)


def show_matrix(table, source, M, col_titles, row_titles, fmt, col_width=45):
    data = {"row": [str(r) for r in row_titles]}
    for j in range(M.shape[1]):
        data[f"c{j}"] = [fmt.format(v) for v in M[:, j]]
    source.data = data
    table.columns = [TableColumn(field="row", title="", width=50)] + [
        TableColumn(field=f"c{j}", title=str(t), width=col_width) for j, t in enumerate(col_titles)
    ]


class_name_input = TextInput(title="Назва класу (напр. '1', '2', '3')", width=200)
add_etalon_button = Button(label="Додати еталон", button_type="primary", width=200)
clear_etalons_button = Button(label="Очистити всі еталони", button_type="warning", width=200)
etalon_controls = row()
etalon_status = Div(text="<i>Завантажте зображення еталону та вкажіть назву класу.</i>", width=900)
etalons_gallery = row()

etalons_source = ColumnDataSource(data=dict(name=[], x=[], xnorm=[], bip=[]))
etalons_table = DataTable(
    source=etalons_source,
    columns=[
        TableColumn(field="name", title="Клас", width=50),
        TableColumn(field="x", title="X (абсолютний вектор)", width=260, formatter=wrap_formatter),
        TableColumn(field="xnorm", title="X_норм (нормований вектор)", width=300, formatter=wrap_formatter),
        TableColumn(field="bip", title="Бінаризований (−1/+1)", width=280, formatter=wrap_formatter),
    ],
    width=900, height=170, row_height=90, autosize_mode="none",
)


def update_etalons_table():
    names, xs, xns, bips = [], [], [], []
    for e in state["etalons"]:
        vec, vn, bip, _, _ = features(e["gray"])
        names.append(e["name"])
        xs.append(str(vec))
        xns.append(str(vn))
        bips.append(str([int(v) for v in bip]))
    etalons_source.data = dict(name=names, x=xs, xnorm=xns, bip=bips)


def redraw_etalons_gallery():
    figs = []
    for e in state["etalons"]:
        f = make_small_figure(e["name"])
        draw_image_with_grid(f, e["gray"])
        figs.append(f)
    etalons_gallery.children = figs


def on_etalon_file_upload(attr, old, new):
    try:
        gray = decode_image(new)
    except Exception as exc:  # noqa: BLE001
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
    if name in [e["name"] for e in state["etalons"]]:
        etalon_status.text = (f"<span style='color:red'>Клас «{html.escape(name)}» уже існує. "
                              f"Введіть іншу назву.</span>")
        return
    state["etalons"].append({"name": name, "gray": gray})
    state["etalon_gray_pending"] = None
    class_name_input.value = ""
    reset_etalon_file_input()
    redraw_etalons_gallery()
    update_etalons_table()
    invalidate()
    etalon_status.text = f"Додано еталон класу «{html.escape(name)}». Всього еталонів: {len(state['etalons'])}."


def clear_etalons():
    state["etalons"].clear()
    state["etalon_gray_pending"] = None
    reset_etalon_file_input()
    redraw_etalons_gallery()
    update_etalons_table()
    invalidate()
    etalon_status.text = "Усі еталони видалено."


reset_etalon_file_input()
add_etalon_button.on_click(add_etalon)
clear_etalons_button.on_click(clear_etalons)

etalons_section = column(
    Div(text="<h3>1. Еталонні образи трьох класів</h3>"),
    etalon_controls,
    clear_etalons_button,
    etalon_status,
    etalons_gallery,
    etalons_table,
)

build_net_button = Button(label="Побудувати матриці мереж", button_type="primary", width=260)
net_status = Div(text="", width=900)
dist_div = Div(text="", width=900)

hop_matrix_source = ColumnDataSource(data=dict(row=[]))
hop_matrix_table = DataTable(source=hop_matrix_source, columns=[], width=900, height=260,
                             autosize_mode="none", index_position=None)
ham_matrix_source = ColumnDataSource(data=dict(row=[]))
ham_matrix_table = DataTable(source=ham_matrix_source, columns=[], width=900, height=260,
                             autosize_mode="none", index_position=None)
hop_matrix_title = Div(text="", width=900)
ham_matrix_title = Div(text="", width=900)


def invalidate(msg=None):
    state["net"] = state["hop"] = state["ham"] = None
    hop_matrix_title.text = ham_matrix_title.text = dist_div.text = ""
    hop_matrix_source.data = dict(row=[])
    ham_matrix_source.data = dict(row=[])
    hop_matrix_table.columns = ham_matrix_table.columns = []
    clear_results()
    net_status.text = msg or ("<i>Матриці ще не побудовано. Додайте еталони і натисніть "
                              "«Побудувати матриці мереж».</i>")


def on_params_change(attr, old, new):
    unknown_vector_div.text = ""
    if state["etalons"]:
        update_etalons_table()
        invalidate("<span style='color:#b36b00'>Параметри змінено — матриці скинуто. "
                   "Натисніть «Побудувати матриці мереж».</span>")


for _w in (rows_spinner, cols_spinner, thresh_spinner):
    _w.on_change("value", on_params_change)
ham_mode_radio.on_change("active", on_params_change)


def distance_table_html(P, names):
    m = len(names)
    D = [[int((P[i] != P[j]).sum()) for j in range(m)] for i in range(m)]
    head = "<tr><th></th>" + "".join(f"<th style='padding:2px 8px'>{html.escape(n)}</th>" for n in names) + "</tr>"
    body = "".join(
        f"<tr><th style='padding:2px 8px'>{html.escape(names[i])}</th>"
        + "".join(f"<td style='padding:2px 8px;text-align:center'>{D[i][j]}</td>" for j in range(m))
        + "</tr>"
        for i in range(m))
    off = [D[i][j] for i in range(m) for j in range(i + 1, m)]
    n = P.shape[1]
    note = ""
    if off:
        d_min = min(off)
        note = (f"<br>Мінімальна відстань між еталонами: <b>{d_min}</b> з {n}. "
                + ("<span style='color:#b36b00'>Вона мала — еталони схожі, тож мережа Хопфілда "
                   "може давати перехресні асоціації.</span>" if d_min < n / 4 else
                   "Еталони достатньо відмінні."))
    return ("<h4>Відстані Хемінга між еталонами</h4>"
            f"<table border='1' style='border-collapse:collapse;font-size:12px'>{head}{body}</table>{note}")


def build_net():
    et = state["etalons"]
    if len(et) < 2:
        net_status.text = "<span style='color:red'>Потрібно щонайменше два еталони.</span>"
        return
    names = [e["name"] for e in et]
    feats = [features(e["gray"]) for e in et]
    P = np.array([f[2] for f in feats])                       
    m, n = P.shape

    W = P.T @ P                                                
    np.fill_diagonal(W, 0)                                     

    mode = ham_mode_radio.active
    if mode == 0:
        W1, b1 = P.T / 2.0, n / 2.0                            
    else:
        W1, b1 = np.array([unit(f[1]) for f in feats]).T, 0.0  
    state["net"] = {"P": P, "W": W, "W1": W1, "b": b1, "mode": mode,
                    "names": names, "n": n, "m": m}

    eps_note = ""
    if not 0 < float(eps_spinner.value) < 1.0 / m:
        eps_spinner.value = round(0.9 / m, 3)
        eps_note = f"<br>ε змінено на {eps_spinner.value:g}, щоб виконувалась умова 0 &lt; ε &lt; 1/m."

    net_status.text = (
        f"Матриці побудовано: n = {n} (сітка {int(rows_spinner.value)}×{int(cols_spinner.value)}), "
        f"m = {m} класів. Ємність мережі Хопфілда ≈ 0,15·n = {0.15 * n:.1f}."
        + ("<br><span style='color:#b36b00'>Увага: кількість образів перевищує ємність мережі Хопфілда.</span>"
           if m > 0.15 * n else "") + eps_note
    )
    hop_matrix_title.text = ("<h4>Мережа Хопфілда: матриця зв'язків W (n×n), "
                             "w<sub>ij</sub> = Σ x<sub>i</sub><sup>k</sup>·x<sub>j</sub><sup>k</sup>, w<sub>ii</sub> = 0</h4>")
    show_matrix(hop_matrix_table, hop_matrix_source, W, range(1, n + 1), range(1, n + 1), "{:.0f}")
    if mode == 0:
        ham_matrix_title.text = ("<h4>Мережа Хемінга: вагові коефіцієнти першого прошарку w<sub>ik</sub> = x<sub>i</sub><sup>k</sup>/2 "
                                 f"(рядки — входи, стовпці — класи), поріг b<sub>k</sub> = n/2 = {n / 2:g}</h4>")
        show_matrix(ham_matrix_table, ham_matrix_source, W1, names, range(1, n + 1), "{:+.1f}", col_width=90)
    else:
        ham_matrix_title.text = ("<h4>Мережа Хемінга: ваги першого прошарку — нормовані еталонні вектори "
                                 "(одиничної довжини), поріг b<sub>k</sub> = 0</h4>")
        show_matrix(ham_matrix_table, ham_matrix_source, W1, names, range(1, n + 1), "{:.3f}", col_width=90)
    dist_div.text = distance_table_html(P, names)
    clear_results()


build_net_button.on_click(build_net)

net_section = column(
    Div(text="<h3>2. Матриці коефіцієнтів мереж</h3>"),
    ham_mode_radio,
    build_net_button,
    net_status,
    dist_div,
    hop_matrix_title,
    hop_matrix_table,
    ham_matrix_title,
    ham_matrix_table,
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
    state["unknown"].update(gray=gray, vector=None, vector_norm=None, bipolar=None)
    unknown_status.text = f"Завантажено ({gray.shape[1]}×{gray.shape[0]} px)."
    draw_image_with_grid(unknown_fig, gray)
    unknown_vector_div.text = ""
    clear_results()  


def build_unknown_vector():
    gray = state["unknown"]["gray"]
    if gray is None:
        unknown_status.text = "<span style='color:red'>Спочатку завантажте невідоме зображення.</span>"
        return False
    vector, vector_norm, bip, rb, cb = features(gray)
    state["unknown"].update(vector=vector, vector_norm=vector_norm, bipolar=bip)
    draw_image_with_grid(unknown_fig, gray, rb, cb)
    unknown_vector_div.text = (
        f"<b>X</b> = ({'; '.join(map(str, vector))})<br>"
        f"<b>X_норм</b> = ({'; '.join(map(str, vector_norm))})<br>"
        f"<b>Бінаризований</b> = ({'; '.join(str(int(v)) for v in bip)})<br>"
        f"{bipolar_grid_html(bip)}"
    )
    return True


unknown_file_input.on_change("value", on_unknown_file_upload)
build_unknown_button.on_click(build_unknown_vector)

unknown_section = column(
    Div(text="<h3>3. Невідомий образ</h3>"),
    row(unknown_file_input, build_unknown_button),
    unknown_status,
    row(unknown_fig, unknown_vector_div),
)

hop_button = Button(label="Класифікувати (Хопфілд)", button_type="success", width=220)
ham_button = Button(label="Класифікувати (Хемінг)", button_type="success", width=220)
classification_div = Div(text="", width=900)

results_source = ColumnDataSource(data=dict(name=[], hop=[], ham1=[], ham2=[]))
results_table = DataTable(
    source=results_source,
    columns=[
        TableColumn(field="name", title="Клас (еталон)"),
        TableColumn(field="hop", title="Хопфілд: відстань Хемінга вихід–еталон"),
        TableColumn(field="ham1", title="Хемінг: y⁽¹⁾ (схожість з еталоном)"),
        TableColumn(field="ham2", title="Хемінг: y⁽²⁾ (другий прошарок)"),
    ],
    width=900, height=130,
)

bar_fig = figure(title="Мережа Хемінга: виходи першого (синій) і другого (зелений) прошарків",
                 width=600, height=250, toolbar_location=None, tools="", x_range=["0"])

hop_iter_title = Div(text="", width=900)
hop_iter_source = ColumnDataSource(data=dict(step=[]))
hop_iter_table = DataTable(source=hop_iter_source, columns=[], width=900, height=260, row_height=28,
                           autosize_mode="none", index_position=None)
ham_iter_title = Div(text="", width=900)
ham_iter_source = ColumnDataSource(data=dict(step=[]))
ham_iter_table = DataTable(source=ham_iter_source, columns=[], width=900, height=260, row_height=28,
                           autosize_mode="none", index_position=None)


def clear_results():
    state["hop"] = state["ham"] = None
    classification_div.text = ""
    results_source.data = dict(name=[], hop=[], ham1=[], ham2=[])
    hop_iter_title.text = ham_iter_title.text = ""
    hop_iter_source.data = dict(step=[])
    ham_iter_source.data = dict(step=[])
    hop_iter_table.columns = ham_iter_table.columns = []
    bar_fig.renderers = []
    bar_fig.x_range.factors = ["0"]


def prepare():
    net = state["net"]
    if net is None:
        classification_div.text = "<span style='color:red'>Спочатку побудуйте матриці мереж.</span>"
        return None
    if not build_unknown_vector():
        classification_div.text = "<span style='color:red'>Спочатку завантажте невідоме зображення.</span>"
        return None
    if len(state["unknown"]["bipolar"]) != net["n"]:
        classification_div.text = "<span style='color:red'>Розмірність вектора не збігається з мережею.</span>"
        return None
    return net


def hopfield_run(W, x, max_iter):
    states, y, status = [x.copy()], x.copy(), "max"
    for _ in range(max_iter):
        s = W @ y
        new = np.where(s > 0, 1, np.where(s < 0, -1, y))
        states.append(new)
        if np.array_equal(new, y):
            status = "stable"
            break
        if len(states) >= 3 and np.array_equal(new, states[-3]):
            status = "cycle"
            break
        y = new
    return states, status


def maxnet_run(y1, eps, max_iter):
    y, hist, status = y1.copy(), [y1.copy()], "max"
    for _ in range(max_iter):
        new = np.clip(y - eps * (y.sum() - y), 0.0, None)
        hist.append(new)
        if np.allclose(new, y):
            status = "stable"
            break
        y = new
    return hist, status


def refresh_results():
    net, hop, ham = state["net"], state["hop"], state["ham"]
    m = net["m"]
    dash = ["—"] * m
    results_source.data = dict(
        name=net["names"],
        hop=[str(d) for d in hop["dist"]] if hop else dash,
        ham1=[f"{v:.4g}" for v in ham["y1"]] if ham else dash,
        ham2=[f"{v:.3f}" for v in ham["y2"]] if ham else dash,
    )
    text = "<br><br>".join(r["verdict"] for r in (hop, ham) if r)
    if hop and ham:
        if hop["label"] is not None and hop["label"] == ham["label"]:
            text += "<br><br><b>Обидві мережі дали однаковий результат.</b>"
        elif hop["label"] is None and ham["label"] is not None:
            text += ("<br><br><b>Хемінг вибрав один клас, а Хопфілд не відновив жодного еталона:</b> "
                     "мережа Хемінга завжди видає номер одного з еталонів, а не хибний образ.")
        else:
            text += "<br><br><b>Мережі дали різні або неоднозначні відповіді.</b>"
    classification_div.text = text


def classify_hopfield():
    net = prepare()
    if net is None:
        return
    x = state["unknown"]["bipolar"]
    P, names, n = net["P"], net["names"], net["n"]
    max_iter = int(iter_spinner.value)
    states, status = hopfield_run(net["W"], x, max_iter)
    final = states[-1]
    dist = [int((final != p).sum()) for p in P]
    k = int(np.argmin(dist))
    negatives = [names[i] for i, p in enumerate(P) if np.array_equal(final, -p)]
    steps = len(states) - 2 if status == "stable" else len(states) - 1

    label = None
    if status != "stable":
        verdict = ("Вихід <b>не стабілізувався</b> "
                   + ("(виник цикл між двома станами)." if status == "cycle" else f"за {max_iter} ітерацій."))
    elif dist[k] == 0:
        label = names[k]
        verdict = f"Мережа відновила еталон класу <b>«{html.escape(names[k])}»</b> — образ розпізнано."
    elif negatives:
        verdict = (f"Мережа збіглася до <b>негативу</b> еталона «{html.escape(negatives[0])}» — "
                   f"хибний образ (відстань до еталона {n} з {n}).")
    else:
        verdict = (f"Мережа збіглася до <b>неіснуючого образу</b>. Найближчий еталон — «{html.escape(names[k])}» "
                   f"(відстань Хемінга {dist[k]} з {n}).")
    state["hop"] = {"label": label, "dist": dist, "verdict":
                    f"<b>Мережа Хопфілда:</b> {verdict} Ітерацій: {steps}.<br>{bipolar_grid_html(final)}"}

    hop_iter_title.text = "<h4>Процес класифікації: мережа Хопфілда</h4>"
    hop_iter_source.data = dict(
        step=["вхід" if t == 0 else f"t = {t}" for t in range(len(states))],
        y=[bipolar_text(s) for s in states],
        changed=[0] + [int((states[t] != states[t - 1]).sum()) for t in range(1, len(states))],
        dist=[str([int((s != p).sum()) for p in P]) for s in states],
    )
    hop_iter_table.columns = [
        TableColumn(field="step", title="Крок", width=70),
        TableColumn(field="y", title="Виходи y (■ = +1, □ = −1)", width=340, formatter=wrap_formatter),
        TableColumn(field="changed", title="Змінилось нейронів", width=130),
        TableColumn(field="dist", title=f"Відстані до еталонів ({', '.join(names)})", width=250),
    ]
    refresh_results()


def classify_hamming():
    net = prepare()
    if net is None:
        return
    names, m = net["names"], net["m"]
    eps = float(eps_spinner.value)
    if not 0 < eps < 1.0 / m:
        classification_div.text = f"<span style='color:red'>Потрібно 0 &lt; ε &lt; 1/m = {1.0 / m:.4f}.</span>"
        return
    if net["mode"] == 0:
        x = state["unknown"]["bipolar"].astype(float)
    else:
        x = unit(state["unknown"]["vector_norm"])
    y1 = net["W1"].T @ x + net["b"]                           
    hist, status = maxnet_run(y1, eps, int(iter_spinner.value))
    y2 = hist[-1]
    alive = np.flatnonzero(y2 > 1e-9)
    steps = len(hist) - 2 if status == "stable" else len(hist) - 1

    label = None
    if status == "stable" and len(alive) == 1:
        label = names[int(alive[0])]
        verdict = f"Залишився один активний нейрон — клас <b>«{html.escape(label)}»</b>. Образ розпізнано."
    else:
        verdict = "Єдиного максимуму не виявлено (виходи другого шару рівні або не стабілізувались)."
    state["ham"] = {"label": label, "y1": [float(v) for v in y1], "y2": [float(v) for v in y2], "verdict":
                    f"<b>Мережа Хемінга:</b> {verdict} Ітерацій другого шару: {steps}; ε = {eps:g}."}

    bar_fig.renderers = []
    bar_fig.x_range.factors = names
    bar_fig.vbar(x=names, top=state["ham"]["y1"], width=0.8, fill_color="#2b8cbe")
    bar_fig.vbar(x=names, top=state["ham"]["y2"], width=0.5, fill_color="#31a354")

    ham_iter_title.text = "<h4>Процес класифікації: мережа Хемінга (другий прошарок)</h4>"
    data = {"step": ["y⁽²⁾(0) = y⁽¹⁾" if t == 0 else f"t = {t}" for t in range(len(hist))]}
    for j in range(m):
        data[f"c{j}"] = [f"{h[j]:.3f}" for h in hist]
    ham_iter_source.data = data
    ham_iter_table.columns = [TableColumn(field="step", title="Крок", width=120)] + [
        TableColumn(field=f"c{j}", title=f"Клас «{nm}»", width=120) for j, nm in enumerate(names)
    ]
    refresh_results()


hop_button.on_click(classify_hopfield)
ham_button.on_click(classify_hamming)

classification_section = column(
    Div(text="<h3>4. Класифікація невідомого образу та аналіз процесу</h3>"),
    row(hop_button, ham_button),
    classification_div,
    results_table,
    bar_fig,
    hop_iter_title,
    hop_iter_table,
    ham_iter_title,
    ham_iter_table,
)

layout = column(
    Div(text="<h2>Лабораторна робота №2, завдання 2. Система розпізнавання на основі мереж Хопфілда та Хемінга</h2>"),
    Div(text="<b>Параметри побудови вектора ознак та мереж:</b>"),
    grid_settings,
    etalons_section,
    net_section,
    unknown_section,
    classification_section,
)

invalidate()
curdoc().add_root(layout)
curdoc().title = "Лабораторна робота №2, завдання 2 — Хопфілд і Хемінг (Bokeh)"