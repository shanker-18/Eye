import re
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.tagging_engine import (
    create_tagged_pdf,
    extract_blocks,
    group_blocks,
    make_doc_id,
    predict_blocks,
    render_page_png,
)


class _NoLifespanProxy:
    def __init__(self, inner: FastAPI) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        else:
            await self._inner(scope, receive, send)


app = _NoLifespanProxy(FastAPI(title="PDF Tagging Engine", version="0.1.0"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_LOCK = Lock()
_FILENAME_TO_DOC_ID: Dict[str, str] = {}


def _safe_stem(filename: str) -> str:
    stem = Path(filename or "document.pdf").stem.strip() or "document"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", stem)


@app.get("/")
def index() -> FileResponse:
    return FileResponse("frontend/index.html")


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.api_route("/.well-known/appspecific/com.chrome.devtools.json", methods=["GET", "HEAD"], include_in_schema=False)
def chrome_devtools_probe() -> Response:
    # Chrome probes this endpoint in dev sessions; return no-content to avoid noisy 404 logs.
    return Response(status_code=204)


@app.get("/api/meta")
def meta() -> Dict[str, Dict[str, str]]:
    return {
        "tagColors": {
            "H1": "rgba(235, 87, 87, 0.45)",
            "H2": "rgba(242, 153, 74, 0.45)",
            "Paragraph": "rgba(47, 128, 237, 0.32)",
            "P": "rgba(47, 128, 237, 0.32)",
            "TOC": "rgba(111, 66, 193, 0.38)",
            "TOCI": "rgba(111, 66, 193, 0.28)",
            "Table": "rgba(39, 174, 96, 0.36)",
            "TR": "rgba(39, 174, 96, 0.30)",
            "TD": "rgba(39, 174, 96, 0.22)",
            "L": "rgba(0, 184, 148, 0.36)",
            "LI": "rgba(0, 184, 148, 0.26)",
            "Reference": "rgba(45, 52, 54, 0.34)",
            "Figure": "rgba(155, 89, 182, 0.40)",
            "Document": "rgba(0, 0, 0, 0.0)",
        }
    }


@app.post("/api/tag-pdf")
async def tag_pdf(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    safe_name = _safe_stem(file.filename)
    doc_id = make_doc_id(pdf_bytes)

    with _CACHE_LOCK:
        cached = _CACHE.get(doc_id)
    if cached is not None:
        with _CACHE_LOCK:
            _FILENAME_TO_DOC_ID[safe_name] = doc_id
        tag_tree = cached["tree"]
        print(tag_tree)
        return JSONResponse(
            {
                "doc_id": doc_id,
                "filename": safe_name,
                "page_count": cached["page_count"],
                "page_sizes": cached["page_sizes"],
                "pages": cached["pages"],
                "blocks": cached["blocks"],
                "tag_tree": tag_tree,
                "tree": tag_tree,
                "cached": True,
            }
        )

    try:
        blocks, page_sizes, page_count = extract_blocks(pdf_bytes)
        predicted_blocks = predict_blocks(pdf_bytes=pdf_bytes, blocks=blocks, model_dir="./model", batch_size=12)
        grouped = group_blocks(predicted_blocks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {exc}") from exc

    payload = {
        "pdf_bytes": pdf_bytes,
        "filename": safe_name,
        "page_count": page_count,
        "page_sizes": page_sizes,
        "pages": grouped["pages"],
        "blocks": grouped["blocks"],
        "tree": grouped["tree"],
        "page_images": {},
    }

    with _CACHE_LOCK:
        _CACHE[doc_id] = payload
        _FILENAME_TO_DOC_ID[safe_name] = doc_id

    tag_tree = grouped["tree"]
    print(tag_tree)

    return JSONResponse(
        {
            "doc_id": doc_id,
            "filename": safe_name,
            "page_count": page_count,
            "page_sizes": page_sizes,
            "pages": grouped["pages"],
            "blocks": grouped["blocks"],
            "tag_tree": tag_tree,
            "tree": tag_tree,
            "cached": False,
        }
    )


@app.get("/api/page-image/{doc_id}/{page_number}")
def page_image(doc_id: str, page_number: int) -> Response:
    with _CACHE_LOCK:
        cached = _CACHE.get(doc_id)

    if cached is None:
        raise HTTPException(status_code=404, detail="Document not found in cache. Re-upload PDF.")

    if page_number < 1 or page_number > cached["page_count"]:
        raise HTTPException(status_code=400, detail="page_number out of range")

    page_key = str(page_number)
    if page_key in cached["page_images"]:
        png_bytes = cached["page_images"][page_key]
    else:
        png_bytes = render_page_png(cached["pdf_bytes"], page_number, zoom=1.8)
        with _CACHE_LOCK:
            cached["page_images"][page_key] = png_bytes

    return Response(content=png_bytes, media_type="image/png")


@app.get("/download-pdf/{filename}")
def download_tagged_pdf(filename: str) -> FileResponse:
    safe_name = _safe_stem(filename)
    with _CACHE_LOCK:
        doc_id = _FILENAME_TO_DOC_ID.get(safe_name)
        cached = _CACHE.get(doc_id) if doc_id else None

    if cached is None:
        raise HTTPException(status_code=404, detail="No tagged document found for filename. Upload PDF first.")

    tagged_name = f"{safe_name}_{doc_id}_tagged.pdf"
    output_path = Path("./outputs") / tagged_name
    create_tagged_pdf(cached["pdf_bytes"], cached["tree"], str(output_path))
    return FileResponse(
        path=str(output_path),
        media_type="application/pdf",
        filename=tagged_name,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


if __name__ == "__main__":
    import uvicorn

    # Disabling lifespan for dev reload avoids noisy CancelledError tracebacks on reload/shutdown.
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True, lifespan="off")
