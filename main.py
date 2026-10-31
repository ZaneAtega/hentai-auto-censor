from base64 import b64decode
from io import BytesIO
from contextlib import asynccontextmanager

from PIL import Image
from ultralytics import YOLO

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response

class Censor:
    def __init__(self):
        self._model = YOLO("models/ntd11_anime_nsfw_segm_v5-variant1.pt")

    def _detect(self, image, conf, iou, device=None, require_both_genitals=False):
        result = self._model.predict(
            source=image,
            conf=conf,
            iou=iou,
            quantize=16,
            device=device,
            classes=[1, 3, 6],
            retina_masks=True,
            channels_last=True
        )[0]

        if result.masks is None or require_both_genitals and (1 not in result.boxes.cls or 3 not in result.boxes.cls):
            return None

        return Image.fromarray(result.masks.data.any(dim=0).cpu().numpy().astype("uint8") * 255)

    def censor(self, image, conf, iou, device=None, require_both_genitals=False, cell_size=None, mask_only=False):
        image = image.convert("RGB")
        mask = self._detect(image, conf, iou, device)

        if mask is None:
            return b""

        output = BytesIO()

        if mask_only:
            mask.save(output, format="PNG")
        else:
            cell_size = max(image.size) // 100 if cell_size is None else int(cell_size)
            small = image.resize((image.width // cell_size, image.height // cell_size))
            image.paste(small.resize(image.size, Image.Resampling.NEAREST), mask)
            image.save(output, format="PNG")

        return output.getvalue()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.censor = Censor()
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/")
async def censor(body: dict):
    encoding = body["image"]
    encoding = encoding.split(",")[1] if encoding.startswith("data:") else encoding

    return Response(app.state.censor.censor(
        Image.open(BytesIO(b64decode(encoding))),
        body.get("conf", 0.5),
        body.get("iou", 1.0),
        body.get("device"),
        body.get("require_both_genitals", True),
        body.get("cell_size"),
        body.get("mask_only")
    ), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7870)