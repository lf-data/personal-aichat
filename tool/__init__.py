
import importlib
import pkgutil
from pathlib import Path
import inspect

def load_tools(*, user_id: int | None = None, **extra_ctx):
    """
    Load user-based tools for the given user ID.
    Args:
        user_id (int): The user ID.
        extra_ctx: Additional context parameters to pass to tool loaders.
    Returns:
        list: A list of user-based tools.
    """
    tools = []
    package_name = __name__
    package_path = Path(__file__).parent

    for module_info in pkgutil.iter_modules([str(package_path)]):
        module_name = module_info.name

        # This ensures we only load tool modules
        if not module_name.endswith("_tool"):
            continue

        full_module_name = f"{package_name}.{module_name}"
        module = importlib.import_module(full_module_name)

        loader = getattr(module, "load_tools", None)
        if not callable(loader):
            continue

        # load kwargs based on function signature
        sig = inspect.signature(loader)
        kwargs = {}

        if "user_id" in sig.parameters:
            kwargs["user_id"] = user_id

        # additional context propagation
        for key, value in extra_ctx.items():
            if key in sig.parameters:
                kwargs[key] = value

        result = loader(**kwargs) if kwargs else loader()

        if result:
            if isinstance(result, list):
                tools.extend(result)
            else:
                # if a single tool is returned, append it
                tools.append(result)

    return tools
