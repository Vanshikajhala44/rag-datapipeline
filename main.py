from rag import RAGPipeline

if __name__ == "__main__":
    rag = RAGPipeline("data/hvm.pdf")
    rag.setup()

    while True:
        query = input("\nAsk something (type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        answer = rag.ask(query)
        print("\nAnswer:\n", answer)