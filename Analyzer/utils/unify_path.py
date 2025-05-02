from pathlib import Path
from typing import Union


def unify_path(path: Union[Path, str]) -> Path:
    """Tries to bring some consistency to paths:
    - Resolve home directories (~ → /home/username).
    - Make paths absolute.

    Args:
        path: The original path.

    Returns:
        The unified path.
    """
    if type(path) == str:
        path = Path(path)

    if str(path).startswith('~'):
        path = path.expanduser()

    if not path.is_absolute():
        file_dir = Path(__file__).parent
        path = file_dir / path

    return path.resolve()