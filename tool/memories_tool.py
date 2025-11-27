from chat_utils.db import (
    add_permanent_memory,
    get_permanent_memories,
    reset_permanent_memories
)
from functools import partial
from langchain.tools import tool
from typing import Annotated


def load_memories_tool(user_id: int):
    """
    Load memory management tools for the given user ID.
    Args:   
        user_id (int): The user ID.
    Returns:
        list: A list of memory management tools.
    """
    # Create partial functions with user_id pre-filled
    partial_add_permanent_memory = partial(add_permanent_memory, user_id = user_id)
    partial_get_permanent_memories = partial(get_permanent_memories, user_id = user_id)
    partial_reset_permanent_memories = partial(reset_permanent_memories, user_id = user_id)

    @tool()
    async def add_memory(memories: Annotated[list[str], "List of memory to add, such as preferences, habits, important dates, etc."]) -> str:
        """
        Add a permanent memory for the user. a permanent memory is a piece of information that the agent should always remember.
        The user provides the memory to add. The memory can be a relevant information about the user, such as preferences, habits, important dates, etc.
        """
        for x in memories:
            await partial_add_permanent_memory(memory=x)
        return "Memories added."

    @tool()
    async def get_memories() -> list[str]:
        """
        Retrieve all permanent memories for the user.
        """
        return await partial_get_permanent_memories()

    @tool()
    async def reset_memories() -> str:
        """
        Reset all permanent memories for the user.
        """
        await partial_reset_permanent_memories()
        return "All memories have been reset."

    return [add_memory, get_memories, reset_memories]


def load_tools(user_id: int):
    """
    Load tools.
    Args:
        user_id (int): The user ID.
    """
    return load_memories_tool(user_id=user_id)