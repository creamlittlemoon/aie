import os
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI

# -----------------------
# 1) Tiny "knowledge base"
# -----------------------
DOCS = [
    ("policy_refund", "Refund policy: You can request a refund within 14 days of purchase if the product is unused."),
    ("policy_shipping", "Shipping policy: Standard shipping takes 3-5 business days. Express shipping takes 1-2 days."),
    ("support_reset", "Password reset: Go to Settings > Security > Reset Password. You will receive a verification email."),
    ("support_invoice", "Invoice: Invoices are available under Billing > Invoices. You can download PDF invoices there."),
]

@dataclass
class Retriever:
    docs: List[Tuple[str, str]]
    vectorizer: TfidfVectorizer
    matrix  : any  # sparse matrix

    @classmethod
    def build(cls, docs: List[Tuple[str, str]]):
        texts = [t for _, t in docs]
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        return cls(docs=docs, vectorizer=vectorizer, matrix=matrix)

    def search(self, query: str, k: int = 2) -> List[Tuple[str, str, float]]:
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix).ravel()
        top_idx = sims.argsort()[::-1][:k]
        return [(self.docs[i][0], self.docs[i][1], float(sims[i])) for i in top_idx]

def build_prompt(question: str, contexts: List[Tuple[str, str, float]]) -> str:
    ctx_block = "\n\n".join(
        [f"[{doc_id} | score={score:.3f}]\n{text}" for doc_id, text, score in contexts]
    )
    return f"""You are a helpful support assistant. Use ONLY the provided context. If the answer is not in context, say "I don't know."

Context:
{ctx_block}

User question: {question}
Answer:"""

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY env var.")

    client = OpenAI(api_key=api_key)

    retriever = Retriever.build(DOCS)

    print("Mini RAG Chatbot (type 'exit' to quit)\n")
    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        hits = retriever.search(q, k=2)
        prompt = build_prompt(q, hits)

        resp = client.chat.completions.create(
            model="gpt-4.1-mini",  # fast + cheap; change if you want
            messages=[
                {"role": "system", "content": "You answer questions using provided context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        ans = resp.choices[0].message.content.strip()
        print(f"\nBot: {ans}\n")

if __name__ == "__main__":
    main()
