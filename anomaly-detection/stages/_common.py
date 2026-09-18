"""Shared scaffolding for the stage scripts."""

import functools
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import anomdet  # noqa: E402

WIDTH = 78


def banner(title, subtitle=""):
    print()
    print("=" * WIDTH)
    print(title)
    if subtitle:
        print(textwrap.fill(subtitle, WIDTH))
    print("=" * WIDTH)
    if anomdet.USING_SOLUTION:
        print("[running against the REFERENCE solution]")


def section(text):
    print(f"\n-- {text} " + "-" * max(0, WIDTH - 4 - len(text)))


def notice(text):
    print()
    print("WHAT TO NOTICE")
    for para in text.strip().split("\n\n"):
        print(textwrap.fill(" ".join(para.split()), WIDTH,
                            initial_indent="  ", subsequent_indent="  "))
        print()


def guard(main):
    """Turn a missing implementation into a useful message, not a traceback."""
    @functools.wraps(main)
    def wrapper():
        try:
            return main()
        except NotImplementedError as e:
            print(f"\n  >>> `{e}` is not implemented yet.")
            print("      Fill it in in anomdet/core.py, then re-run this stage.")
            print("      To see the finished output first:")
            print(f"        ANOMDET_USE_REFERENCE=1 python3 {sys.argv[0]}\n")
            return 1
    return wrapper
