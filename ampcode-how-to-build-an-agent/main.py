from anthropic import Anthropic


class Agent:
    def __init__(self, client, get_user_message):
        self.client = client
        self.get_user_message = get_user_message

    def run(self):
        conversation = []

        print("Chat with Claude (use 'ctrl-c' to quit)")

        while True:
            print("\033[94mYou\033[0m: ", end="", flush=True)
            user_input, ok = self.get_user_message()
            if not ok:
                break

            user_message = {
                "role": "user",
                "content": [{"type": "text", "text": user_input}],
            }
            conversation.append(user_message)

            message = self.run_inference(conversation)
            assistant_text_parts = []
            for content in message.content:
                if content.type == "text":
                    assistant_text_parts.append(content.text)
                    print(f"\033[93mClaude\033[0m: {content.text}")

            conversation.append(
                {
                    "role": "assistant",
                    "content": "".join(assistant_text_parts),
                }
            )
        return None

    def run_inference(self, conversation):
        return self.client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            messages=conversation,
        )


def main():
    client = Anthropic()

    def get_user_message():
        try:
            line = input()
            return line, True
        except EOFError:
            return "", False

    agent = Agent(client, get_user_message)

    try:
        agent.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()