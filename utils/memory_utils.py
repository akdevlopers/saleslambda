import os
import json
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema.messages import messages_from_dict,messages_to_dict

def get_memory_path(session_id):
    return f"faiss_memories/{session_id}.json"

def load_persistent_memory(session_id, llm):
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=1000,
        return_messages=True,
        memory_key="chat_history"
    )

    path = get_memory_path(session_id)

    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)

            message_dicts = data.get("messages", [])

            #  Properly convert message dictionaries to LangChain message objects
            memory.chat_memory.messages = messages_from_dict(message_dicts)
            print(f"Loaded memory from {path} with {len(message_dicts)} messages")

        except Exception as e:
            print(f"Failed to load memory: {e}")
    else:
        print(f"No memory file found at {path}, starting fresh.")

    return memory


def save_persistent_memory(session_id, memory):
    path = get_memory_path(session_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(
            {"messages": messages_to_dict(memory.chat_memory.messages)},
            f,
            indent=2
        )
    print(f"Memory saved to {path}")
