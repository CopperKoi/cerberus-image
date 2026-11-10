"""Offline tool smoke. It tests availability, not benchmark solve ability."""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    results: dict[str, dict[str, bool]] = {}

    def check(domain: str, name: str, ok: bool) -> None:
        results.setdefault(domain, {})[name] = ok

    for domain, programs in {
        "web": ["curl", "nmap"],
        "pwn": ["gcc", "gdb", "objdump"],
        "reverse": ["objdump", "strings"],
        "misc": ["file", "7z", "exiftool", "tshark"],
    }.items():
        for name in programs:
            check(domain, name, shutil.which(name) is not None)

    for domain, modules in {
        "web": ["httpx", "bs4"],
        "pwn": ["pwn", "elftools"],
        "reverse": ["capstone", "unicorn", "z3"],
        "crypto": ["Crypto", "sympy", "z3"],
        "misc": ["PIL", "scapy"],
        "ai_security": ["httpx", "json"],
    }.items():
        for name in modules:
            try:
                importlib.import_module(name)
                ok = True
            except (ImportError, OSError):
                ok = False
            check(domain, name, ok)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "hello.c"
        target = root / "hello"
        source.write_text("int main(void) { return 7; }\n", encoding="utf-8")
        compile_result = subprocess.run(
            ["gcc", "-o", str(target), str(source)], capture_output=True, check=False,
        )
        check("pwn", "compile_elf", compile_result.returncode == 0 and target.is_file())
        if target.is_file():
            inspect = subprocess.run(
                ["objdump", "-f", str(target)], capture_output=True, check=False,
            )
            check("reverse", "inspect_elf", inspect.returncode == 0 and b"elf64-x86-64" in inspect.stdout)

    print(json.dumps({"schema_version": 1, "offline": True, "capabilities": results}, sort_keys=True))
    return 0 if all(all(values.values()) for values in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
