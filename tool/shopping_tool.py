from chat_utils.db import (
    get_shopping_list,
    add_or_update_shopping_list,
    delete_shopping_list,
)
from functools import partial
from langchain.tools import tool
from typing import Annotated, Literal
from pydantic import BaseModel
from uuid import uuid4

class ShoppingItem(BaseModel):
    """
    Model representing a shopping item.
    """
    product_name: Annotated[str, "The product name in snake case"]
    quantity: Annotated[int, "The amount of product to add"]
    metric: Annotated[
        Literal["kg", "g", "ml", "l", "pacco", "pezzo"],
            "The metric to measure the quantity of product",
    ]



def load_shopping_tool(user_id: int):
    """
    Load shopping management tools for the given user ID.
    Args:
        user_id (int): The user ID.
    Returns:
        list: A list of shopping management tools.
    """

    # Create partial functions with user_id pre-filled
    partial_get_shopping_list = partial(get_shopping_list, user_id=user_id)
    partial_add_or_update_shopping_list = partial(
        add_or_update_shopping_list, user_id=user_id
    )
    partial_delete_shopping_list = partial(delete_shopping_list, user_id=user_id)

    @tool()
    async def get_product_list() -> str:
        """
        Retrieve all product to buy in shop.
        """
        return await partial_get_shopping_list()

    @tool()
    async def add_or_update_product(
        list_product: Annotated[list[ShoppingItem], "List of product to add or update"],
    ) -> list[str]:
        """
        Add a product to buy into a shop with quantity and metric.
        """
        for x in list_product:
            await partial_add_or_update_shopping_list(
                product_id=str(uuid4()), list_name=x.product_name, quantity=x.quantity, metric=x.metric
            )
        return "products are added"

    @tool()
    async def delete_product(list_product_id: Annotated[list[str], "The list of product id to remove in snake case"]) -> str:
        """
        Reset all permanent memories for the user.
        """
        for x in list_product_id:
            await partial_delete_shopping_list(product_id=x)
        return "The products has been deleted"

    return [get_product_list, add_or_update_product, delete_product]


def load_tools(user_id: int):
    """
    Load tools.
    Args:
        user_id (int): The user ID.
    """
    return load_shopping_tool(user_id=user_id)
