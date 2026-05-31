from dotenv import load_dotenv

load_dotenv()

from graph.graph_app import app

if __name__ == "__main__":
    print("Hello Adaptive RAG")
    print(app.invoke(input={"question": "what is the agent memory?"}))