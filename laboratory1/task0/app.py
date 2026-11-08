import base64
from io import BytesIO

from PIL import Image

from bokeh.layouts import column, row
from bokeh.models import FileInput, Div
from bokeh.plotting import curdoc, figure

info_div = Div(text="<b>Будь ласка, завантажте зображення (.bmp, .png, .jpg).</b>", width=500)

p = figure(
    title="Завантажене зображення",
    x_range=(0, 100), y_range=(0, 100),
    width=400, height=450,
    toolbar_location=None, min_border=0,
)
p.axis.visible = False
p.grid.visible = False


def process_image(attr, old, new):
    try:
        raw_bytes = base64.b64decode(file_input.value)
        img = Image.open(BytesIO(raw_bytes)).convert('RGBA')

        img_width, img_height = img.size
        
        import numpy as np
        img_np = np.array(img, dtype=np.uint8)
        img_uint32 = np.empty((img_height, img_width), dtype=np.uint32)
        view = img_uint32.view(dtype=np.uint8).reshape((img_height, img_width, 4))
        view[:, :, :] = img_np
        img_uint32 = np.flipud(img_uint32)

        p.x_range.start = 0
        p.x_range.end = img_width
        p.y_range.start = 0
        p.y_range.end = img_height

        p.renderers = []
        p.image_rgba(image=[img_uint32], x=0, y=0, dw=img_width, dh=img_height)

        info_div.text = f"<b>Статус:</b> Успішно завантажено. Розмір: {img_width}×{img_height} пікселів."

    except Exception as e:
        info_div.text = f"<b>Помилка:</b> {str(e)}"


file_input = FileInput(accept=".bmp,.png,.jpg,.jpeg")
file_input.on_change('value', process_image)

layout = column(
    Div(text="<h2>Перегляд завантажених зображень</h2>"),
    info_div,
    file_input,
    row(column(Div(text="<h3>Зображення</h3>"), p)),
)

curdoc().add_root(layout)
curdoc().title = "Перегляд зображень - Bokeh"