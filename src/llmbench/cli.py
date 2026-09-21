"""The ``llmbench`` command. See ``containers/cli.py`` for the commands themselves."""

from .containers.cli import main, parser

__all__ = ["main", "parser"]

if __name__ == "__main__":
    raise SystemExit(main())
