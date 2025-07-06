import gradio as gr
import chromadb
from chromadb.utils import embedding_functions
from transformers import pipeline

# Load vector store and model
chroma_client = chromadb.PersistentClient(path="vector_store/chroma")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(name="complaints", embedding_function=embedding_fn)
rag_generator = pipeline("text-generation", model="distilgpt2", max_new_tokens=200)

# Prompt builder
def build_prompt(context, question):
    return f"""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, say you don't have enough information.

Context:
{context}

Question: {question}

Answer:
"""

# RAG function
def ask_question(question, top_k=5):
    results = collection.query(query_texts=[question], n_results=top_k)
    retrieved_chunks = results['documents'][0]
    context = "\n---\n".join(retrieved_chunks)
    prompt = build_prompt(context, question)
    response = rag_generator(prompt, do_sample=True)[0]['generated_text']
    answer = response[len(prompt):].strip()
    sources = "\n\n".join([f"• {c}" for c in retrieved_chunks])
    return answer, sources

# Gradio UI
with gr.Blocks(css="""
#input-box textarea {
    height: 80px !important;
    font-size: 14px;
    padding: 8px;
}
#ask-btn, #clear-btn {
    background-color: #8B4513 !important;
    color: white !important;
    font-weight: bold;
    border-radius: 6px;
    padding: 6px 14px;
    margin: 4px;
}
.gradio-container {
    max-width: 800px !important;
    margin: auto !important;
}
h1, h2 {
    color: #1E90FF !important;  /* DodgerBlue */
}
""") as demo:

    # Title and subtitle in blue
    gr.Markdown("# 10 Academy: Artificial Intelligence Mastery")
    gr.Markdown("## CrediTrust Complaint Assistant")

    gr.Markdown("Ask a question about customer complaints across financial products.")

    user_input = gr.Textbox(
        lines=3,
        placeholder="Type your question here...",
        label="Your Question",
        show_label=True,
        elem_id="input-box"
    )

    with gr.Row():
        ask_button = gr.Button("Ask", elem_id="ask-btn")
        clear_button = gr.Button("Clear", elem_id="clear-btn")

    answer_output = gr.Textbox(label="AI Answer", lines=5)
    sources_output = gr.Textbox(label="Retrieved Complaint Sources", lines=7)

    ask_button.click(fn=ask_question, inputs=[user_input], outputs=[answer_output, sources_output])
    clear_button.click(fn=lambda: ("", ""), inputs=[], outputs=[answer_output, sources_output])

if __name__ == "__main__":
    demo.launch()
