from ollama import chat
from ollama import ChatResponse


def main():
    prompt = input()
    while prompt.strip().lower() != 'exit':
        print('Generation response...')
        response = chat(
        model='qwen3:1.7b',
            messages=[
                {'role': 'user', 'content': prompt}
            ],
        )
        print(response.message.content)
        prompt = input()


if __name__ == "__main__":
    main()
