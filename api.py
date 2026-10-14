import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import asyncio
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import io

from predict import load_predictor, predict_image


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DEFAULT_WEIGHTS = BASE_DIR / "weights" / "best_resnet18.pth"

app = FastAPI(
    title="猫狗分类 API",
    description="Cat / Dog Image Classifier - FastAPI Backend",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor = None
_predictor_meta = {}


def get_predictor():
    global _predictor, _predictor_meta
    if _predictor is None:
        weight_path = os.environ.get("MODEL_WEIGHTS", str(DEFAULT_WEIGHTS))
        model_name = os.environ.get("MODEL_NAME", "resnet18")
        device = os.environ.get("DEVICE", "cpu")
        _predictor, device = load_predictor(weight_path, model_name, device)
        _predictor_meta = {"weights": weight_path, "model": model_name, "device": str(device)}
    return _predictor


@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"service": "cat-dog-classifier", "status": "ok"}


@app.get("/api/health")
async def health():
    get_predictor()
    return {"status": "ok", "model": _predictor_meta}


@app.get("/api/models")
async def list_models():
    weights_dir = BASE_DIR / "weights"
    available = []
    if weights_dir.exists():
        for f in sorted(weights_dir.iterdir()):
            if f.suffix == ".pth":
                available.append(f.name)
    return {"available_weights": available, "default": str(DEFAULT_WEIGHTS.name)}


@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    predictor = get_predictor()

    suffix = Path(file.filename or "img.jpg").suffix.lower()
    if suffix and suffix not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {suffix}")

    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法解析图片: {e}")

    result = predict_image(img, predictor, _predictor_meta.get("device", "cpu"))
    result["filename"] = file.filename
    result["model_info"] = _predictor_meta
    return result


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    parser = argparse.ArgumentParser(description="启动 FastAPI 后端")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                        help="模型权重 .pth 路径")
    parser.add_argument("--model", default="resnet18",
                        help="模型名称 (resnet18 / alexnet_tv ...)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.environ["MODEL_WEIGHTS"] = args.weights
    os.environ["MODEL_NAME"] = args.model
    os.environ["DEVICE"] = args.device

    get_predictor()
    print(f"\n🚀 后端已就绪 -> http://{args.host}:{args.port}")
    print(f"   前端页面:  http://127.0.0.1:{args.port}/")
    print(f"   API 文档:  http://127.0.0.1:{args.port}/docs")
    print(f"   模型: {_predictor_meta}\n")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
