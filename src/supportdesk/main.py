from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(
        title="SupportDesk RAG",
        version="0.1.0",
        summary="Synthetic Northline Telecom support agent.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
