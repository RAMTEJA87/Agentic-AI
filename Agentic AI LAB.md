Agentic AI Lab – Exercises 1 to 6
Python: 3.10–3.12  
LLM: Groq (`openai/gpt-oss-20b`)
1. Common Setup
Project structure
```text
AGENTIC_AI/
├── .env
├── requirements.txt
├── exercise1/
│   └── main.py
├── exercise2/
│   └── main.py
├── exercise3/
│   ├── main.py
│   └── data/
│       └── syllabus.txt
├── exercise4/
│   └── main.py
├── exercise5/
│   ├── main.py
│   └── data/
│       ├── handbook.txt
│       ├── exam_rules.txt
│       └── department.txt
└── exercise6/
    └── main.py
```
Create and activate environment
Windows PowerShell
```powershell
mkdir AGENTIC_AI
cd AGENTIC_AI
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```
`requirements.txt`
```text
groq
python-dotenv
langchain
langchain-community
langchain-text-splitters
langchain-groq
langchain-huggingface
langchain-chroma
sentence-transformers
chromadb
pypdf
langgraph
```
Install packages:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
`.env`
Create a Groq API key at https://console.groq.com/keys, then save it in `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
```
Do not share or upload the `.env` file. Never put the API key directly in Python code.
---
Exercise 1 – Campus Query Router Agent
File structure
```text
exercise1/
├── .env
├── requirements.txt
└── main.py
```
`main.py`
```python
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def academic_tool(query):
    return f"Academic information selected for: {query}"

def exam_tool(query):
    return f"Exam information selected for: {query}"

def general_tool(query):
    return f"General information selected for: {query}"

tools = {
    "academic": academic_tool,
    "exam": exam_tool,
    "general": general_tool
}

def classify_query(query):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "Classify the query as exactly one label: academic, exam, or general. Academic means subjects, syllabus, classes, attendance, assignments or courses. Exam means exam dates, hall tickets, results, marks or regulations. General means all other campus queries. Return only the label."
            },
            {"role": "user", "content": query}
        ],
        temperature=0
    )
    label = response.choices[0].message.content.strip().lower()
    return label if label in tools else "general"

def run_agent(query):
    intent = classify_query(query)
    print("Selected intent:", intent)
    print("Tool response:", tools[intent](query))

def main():
    while True:
        query = input("\nEnter campus query (exit to stop): ").strip()
        if query.lower() == "exit":
            break
        if query:
            try:
                run_agent(query)
            except Exception as error:
                print("Agent error:", error)

if __name__ == "__main__":
    main()
```
Run
```powershell
python exercise1/main.py
```
Sample output
```text
Enter campus query (exit to stop): When is the semester examination?
Selected intent: exam
Tool response: Exam information selected for: When is the semester examination?
```
---
Exercise 2 – Student Helper Agent
File structure
```text
exercise2/
├── .env
├── requirements.txt
└── main.py
```
`main.py`
```python
import os
import ast
import operator
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos
}

def calculate_node(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](
            calculate_node(node.left),
            calculate_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](calculate_node(node.operand))
    raise ValueError("Only basic arithmetic expressions are allowed.")

def math_tool(query):
    expression = query.replace("×", "*").replace("÷", "/")
    tree = ast.parse(expression, mode="eval")
    result = calculate_node(tree.body)
    return f"Result: {result}"

def python_help_tool(query):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are a Python tutor. Explain the solution clearly and provide a short correct Python example."},
            {"role": "user", "content": query}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def english_grammar_tool(query):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "Correct the grammar of the user's text. Return the corrected text and nothing else."},
            {"role": "user", "content": query}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

tools = {
    "math": math_tool,
    "python_help": python_help_tool,
    "english_grammar": english_grammar_tool
}

def classify_query(query):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "Choose exactly one tool label: math, python_help, english_grammar. Use math for calculations, python_help for Python programming questions, and english_grammar for sentence correction. Return only the label."
            },
            {"role": "user", "content": query}
        ],
        temperature=0
    )
    label = response.choices[0].message.content.strip().lower()
    if label not in tools:
        raise ValueError(f"Unexpected tool label: {label}")
    return label

def run_agent(query):
    tool_name = classify_query(query)
    print("Selected tool:", tool_name)
    approval = input("Execute this tool? (y/n): ").strip().lower()
    if approval != "y":
        print("Execution cancelled.")
        return
    try:
        print("Tool response:", tools[tool_name](query))
    except Exception as error:
        print("Tool error:", error)

def main():
    while True:
        query = input("\nAsk Student Helper (exit to stop): ").strip()
        if query.lower() == "exit":
            break
        if query:
            try:
                run_agent(query)
            except Exception as error:
                print("Agent error:", error)

if __name__ == "__main__":
    main()
```
Run
```powershell
python exercise2/main.py
```
Sample output
```text
Ask Student Helper (exit to stop): 25 * 8
Selected tool: math
Execute this tool? (y/n): y
Tool response: Result: 200
```
---
Exercise 3 – LangChain Simple Retrieval Agent
File structure
```text
exercise3/
├── .env
├── requirements.txt
├── main.py
└── data/
    └── syllabus.txt
```
`data/syllabus.txt`
Create the file and add your syllabus content. Example:
```text
The Computer Science Engineering programme includes programming, data structures, database management systems, operating systems, computer networks and artificial intelligence.
Students must maintain the attendance percentage prescribed by the college.
Internal assessment and semester-end examinations are part of the evaluation process.
Students should complete laboratory records and assignments before the deadlines announced by faculty.
```
`main.py`
```python
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing from the root .env file.")

source_file = BASE_DIR / "data" / "syllabus.txt"
if not source_file.exists():
    raise FileNotFoundError(f"Add the syllabus text file at: {source_file}")

documents = TextLoader(str(source_file), encoding="utf-8").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="syllabus_retrieval",
    persist_directory=str(BASE_DIR / "chroma_db")
)

retriever = db.as_retriever(search_kwargs={"k": 3})
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=api_key)

prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context below. "
    "If the answer is not present, say 'Information not found in the syllabus.'\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

chain = prompt | llm

def main():
    print("Syllabus Retrieval Agent. Type exit to stop.")
    while True:
        question = input("\nAsk a syllabus question: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        results = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in results)
        answer = chain.invoke({"context": context, "question": question})
        print("\nAnswer:\n", answer.content)
        print("\nRetrieved sources:")
        for doc in results:
            print("-", doc.metadata.get("source", "syllabus.txt"))
            print(doc.page_content)

if __name__ == "__main__":
    main()
```
Run
```powershell
python exercise3/main.py
```
Sample output
```text
Syllabus Retrieval Agent. Type exit to stop.
Ask a syllabus question: Which subjects are included in the CSE programme?

Answer:
 The CSE programme includes programming, data structures, database management systems, operating systems, computer networks and artificial intelligence.

Retrieved sources:
- .../syllabus.txt
The Computer Science Engineering programme includes programming, data structures, database management systems, operating systems, computer networks and artificial intelligence.
```
---
Exercise 4 – LangGraph Self-Correction Agent
File structure
```text
exercise4/
├── .env
├── requirements.txt
└── main.py
```
`main.py`
```python
import os
import ast
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing from .env")

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=api_key)

class AgentState(TypedDict):
    question: str
    code: str
    error: str
    attempts: int

def generate_code(state: AgentState):
    response = llm.invoke(
        "Write a complete, simple Python solution. Return Python code only.\nTask: "
        + state["question"]
    )
    return {"code": response.content.strip(), "attempts": 1}

def validate_code(state: AgentState):
    code = state["code"].replace("
```python", "").replace("```", "").strip()
    try:
        ast.parse(code)
        return {"code": code, "error": ""}
    except SyntaxError as error:
        return {"code": code, "error": str(error)}

def correct_code(state: AgentState):
    response = llm.invoke(
        "Fix the Python syntax error. Return only complete Python code.\n"
        f"Task: {state['question']}\nError: {state['error']}\nCode:\n{state['code']}"
    )
    return {
        "code": response.content.replace("```python", "").replace("```", "").strip(),
        "attempts": state["attempts"] + 1
    }

def route_after_validation(state: AgentState):
    if state["error"] and state["attempts"] < 2:
        return "correct"
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("generate", generate_code)
workflow.add_node("validate", validate_code)
workflow.add_node("correct", correct_code)
workflow.add_edge(START, "generate")
workflow.add_edge("generate", "validate")
workflow.add_conditional_edges(
    "validate",
    route_after_validation,
    {"correct": "correct", "end": END}
)
workflow.add_edge("correct", "validate")
app = workflow.compile()

def main():
    question = input("Enter a Python coding task: ").strip()
    if not question:
        return
    result = app.invoke({
        "question": question,
        "code": "",
        "error": "",
        "attempts": 0
    })
    print("\nFinal code:\n")
    print(result["code"])
    print("\nSyntax validation:", "Passed" if not result["error"] else result["error"])
    print("Generation attempts:", result["attempts"])

if __name__ == "__main__":
    main()
```
Run
```powershell
python exercise4/main.py
```
Sample output
```text
Enter a Python coding task: Write a Python program to reverse a list.

Final code:

def reverse_list(items):
    return items[::-1]

Syntax validation: Passed
Generation attempts: 1
```
---
Exercise 5 – Multi-Document Campus RAG Agent
File structure
```text
exercise5/
├── .env
├── requirements.txt
├── main.py
└── data/
    ├── handbook.txt
    ├── exam_rules.txt
    └── department.txt
```
`data/handbook.txt`

```text
Students should follow the academic calendar published by the college.
Students must maintain the attendance percentage required by college regulations.
Assignments and laboratory records must be submitted within the deadlines announced by faculty.
```
`data/exam_rules.txt`
```text
Students must carry their hall ticket and valid college identity card to the examination hall.
Electronic devices are not permitted inside the examination hall unless explicitly authorized.
Students should follow the examination timetable announced by the examination branch.
```
`data/department.txt`
```text
The Computer Science and Engineering department offers courses in programming, data structures, databases, operating systems, computer networks and artificial intelligence.
The department encourages students to participate in projects, internships, technical events and workshops.
```
`main.py`
```python
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing from the root .env file.")

data_dir = BASE_DIR / "data"
file_names = ["handbook.txt", "exam_rules.txt", "department.txt"]
documents = []

for file_name in file_names:
    file_path = data_dir / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Missing document: {file_path}")
    documents.extend(TextLoader(str(file_path), encoding="utf-8").load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="campus_information",
    persist_directory=str(BASE_DIR / "chroma_db")
)

retriever = db.as_retriever(search_kwargs={"k": 4})
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=api_key)

prompt = ChatPromptTemplate.from_template(
    "Answer only from the supplied campus document context. "
    "Do not invent rules or details. If the answer is missing, say "
    "'Information not found in the provided documents.'\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

chain = prompt | llm

def main():
    print("Campus RAG Agent. Type exit to stop.")
    while True:
        question = input("\nAsk a campus question: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        results = retriever.invoke(question)
        context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for doc in results
        )
        response = chain.invoke({"context": context, "question": question})
        print("\nAnswer:\n", response.content)
        print("\nReference snippets:")
        for doc in results:
            print(f"\nSource: {Path(doc.metadata.get('source', 'unknown')).name}")
            print(doc.page_content)

if __name__ == "__main__":
    main()
```
Run
```powershell
python exercise5/main.py
```
Sample output
```text
Campus RAG Agent. Type exit to stop.
Ask a campus question: What should students carry to the examination hall?

Answer:
 Students must carry their hall ticket and valid college identity card.

Reference snippets:

Source: exam_rules.txt
Students must carry their hall ticket and valid college identity card to the examination hall.
```
---
Exercise 6 – Two-Agent Collaboration
File structure
```text
exercise6/
├── .env
├── requirements.txt
└── main.py
```
`main.py`
```python
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing from .env")

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=api_key)

def research_agent(topic):
    prompt = (
        "You are a Research Agent. Generate 5 accurate, concise bullet points "
        f"about the following topic:\n{topic}"
    )
    return llm.invoke(prompt).content.strip()

def refiner_agent(topic, points):
    prompt = (
        "You are a Refiner Agent. Convert the research points into a short, "
        "well-structured explanation. Do not add unsupported facts.\n"
        f"Topic: {topic}\nResearch points:\n{points}"
    )
    return llm.invoke(prompt).content.strip()

def main():
    topic = input("Enter a topic: ").strip()
    if not topic:
        return
    print("\nResearch Agent:")
    points = research_agent(topic)
    print(points)
    print("\nRefiner Agent:")
    answer = refiner_agent(topic, points)
    print("\nFinal explanation:\n")
    print(answer)

if __name__ == "__main__":
    main()
```
Run
```powershell
python exercise6/main.py
```
Sample output
```text
Enter a topic: Applications of Artificial Intelligence in education

Research Agent:
- Personalized learning
- Automated assessment
- Intelligent tutoring systems
- Learning analytics
- Content generation

Refiner Agent:

Final explanation:

Artificial Intelligence supports education through personalized learning,
automated assessment, intelligent tutoring, learning analytics and content
generation. These applications can help teachers understand student progress
and provide learning support.
```
---
