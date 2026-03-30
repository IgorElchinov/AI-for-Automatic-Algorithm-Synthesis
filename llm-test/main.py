from ollama import chat
from ollama import ChatResponse


def main():
    response = chat(
    model='qwen3:1.7b',
        messages=[
            {'role': 'user', 'content': 'Give me 3 bullet points about local LLM APIs.'}
        ],
    )

    print(response.message.content)


if __name__ == "__main__":
    main()
