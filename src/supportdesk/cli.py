def main() -> None:
    print("SupportDesk RAG")
    print("  python -m alembic upgrade head")
    print("  python -m supportdesk.ingest")
    print("  python -m supportdesk.manifest")
    print("  python -m supportdesk.eval")
    print("  python -m supportdesk.token <customer|agent|finance>")
    print("API: python -m uvicorn supportdesk.main:app --reload")
