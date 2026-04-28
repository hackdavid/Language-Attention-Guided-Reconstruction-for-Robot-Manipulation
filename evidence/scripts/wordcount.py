"""Count words in evidence/report2.md per section, excluding headings, tables and code blocks."""
import re
import sys
from pathlib import Path

p = Path(__file__).resolve().parents[2] / "evidence" / "report2.md"
text = p.read_text(encoding="utf-8")

m1 = re.search(r"## Abstract", text)
m2 = re.search(r"^## References", text, re.MULTILINE)
body = text[m1.start():m2.start()]

def count(s: str) -> int:
    s = re.sub(r"^\|.*\|\s*$", "", s, flags=re.MULTILINE)  # tables
    s = re.sub(r"```.*?```", "", s, flags=re.DOTALL)        # code blocks
    s = re.sub(r"^#+ .*$", "", s, flags=re.MULTILINE)        # headings
    return len(re.findall(r"[A-Za-z0-9'-]+", s))

print("Total body (Abstract -> end of Conclusion, excluding headings, tables, code):",
      count(body))
print()
parts = re.split(r"^## ", body, flags=re.MULTILINE)
for part in parts:
    if not part.strip():
        continue
    title = part.split("\n", 1)[0].strip()[:60]
    n = count("## " + part)
    print(f"  {n:5d}  {title!r}")
