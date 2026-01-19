from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from app.rag_engine import rag_engine
from app.schemas import AskRequest, AskResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_engine.initialize()
    yield
    rag_engine.refresh()

app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
async def health_check():
    stats = rag_engine.get_stats()
    return {"status": "healthy", **stats}


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    result = rag_engine.ask(request.question, tables=request.tables)
    return AskResponse(**result)
