from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src.indexing import build_vector_store

# Initialize components
vector_store = build_vector_store()
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Prompt template
prompt = PromptTemplate(
    template="""
You are a helpful medical coding assistant.
Answer ONLY from the provided medical coding question context, and explain why you think that is the answer.

{context}
Question: {question}
Options: {options}
""",
    input_variables=["context", "question", "options"]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def make_query(inputs):
    return f"Q: {inputs['question']}\n{inputs['options']}"

retrieval_chain = RunnableParallel({
    "context": RunnableLambda(make_query) | retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
    "options": RunnablePassthrough()
})

full_chain = retrieval_chain | prompt | llm | StrOutputParser()

def run_pipeline(question, options):
    return full_chain.invoke({"question": question, "options": options})