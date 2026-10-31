"""Grounding check ko jaanbujhkar galat answer do. Expected: grounded=False."""
from langchain_core.documents import Document

from app.nodes import check_grounding

CONTEXT = Document(
    page_content=(
        "## Data validation\n\nIf you go to /items/foo you will see an HTTP error: "
        '{"detail": [{"type": "int_parsing", "msg": "Input should be a valid integer"}]} '
        "because the path parameter item_id had a value of foo, which is not an int."
    ),
    metadata={"source": "demo.md", "section": "Data validation"},
)

CASES = {
    "SUPPORTED (expect True)": "Passing /items/foo to an int path parameter gives an HTTP error with type int_parsing and message 'Input should be a valid integer' [1].",
    "UNSUPPORTED FIX (expect False)": "It gives an int_parsing error [1]. To fix it, change the annotation to str so FastAPI accepts any value [1].",
    "UNSUPPORTED FACT (expect False)": "It gives an int_parsing error [1]. FastAPI also retries the request three times automatically [1].",
}

for name, answer in CASES.items():
    out = check_grounding({"relevant_documents": [CONTEXT], "answer": answer})
    print(f"{name}\n   grounded={out['grounded']}  unsupported={out['grounding_feedback'] or '-'}\n")