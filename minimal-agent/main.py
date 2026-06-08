import re
import subprocess
import os
from litellm import completion

env_vars = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}

def query_lm(messages: list[dict[str, str]]) -> str:
    response = completion(
        model="openai/gpt-5.4-nano",
        messages=messages
    )
    return response.choices[0].message.content

def parse_action(lm_output: str) -> str:
    """Take LM output, return action"""
    matches = re.findall(
        r"```bash-action\s*\n(.*?)\n```", 
        lm_output, 
        re.DOTALL
    )
    return matches[0].strip() if matches else ""

def execute_action(command: str) -> str:
    """Execute action, return output"""
    result = subprocess.run(
        command,
        shell=True,
        text=True,
        env=os.environ | env_vars,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    return result.stdout

# Main agent loop
messages = [{
    "role": "system", 
    "content": "You are an helpful planner assistant. When you want to run a command, wrap it in ```bash-action\n<command>\n```. Give only 1 commmand. To finish, run the exit command. You can ask clarifying questions till an agreement is reached if the approach or current state is not clear."
}, {
    "role": "user", 
    "content": "I need to create a chat bar that embeds itself into a website and help users ask intelligent questions and customize the wbesite for their taste/understanding. The sell for businesses it it will help me increase their conversion rates when people discover their website or try to understand it."
}]

if __name__ == "__main__":
    while True:
        lm_output = query_lm(messages)
        print("LM output", lm_output)
        messages.append({"role": "assistant", "content": lm_output})  # remember what the LM said
        action = parse_action(lm_output)  # separate the action from output
        print("Action", action)
        if action == "exit":
            break
        output = execute_action(action)
        print("Output", output)
        messages.append({"role": "user", "content": output})  # send command output back
        break