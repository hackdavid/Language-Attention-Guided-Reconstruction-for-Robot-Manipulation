"""Verify that every citation in the body has a matching reference and vice versa."""
import re
from pathlib import Path

p = Path(__file__).resolve().parents[2] / "evidence" / "report2.md"
text = p.read_text(encoding="utf-8")

m_refs = re.search(r"^## References\s*$", text, re.MULTILINE)
m_ai = re.search(r"^## Use of AI Tools", text, re.MULTILINE)
body = text[: m_refs.start()]
refs_section = text[m_refs.start() : m_ai.start()]

cite_in_body = sorted({int(n) for n in re.findall(r"\[(\d+)\]", body)})
cite_in_refs = sorted({int(n) for n in re.findall(r"^\[(\d+)\]", refs_section, re.MULTILINE)})

print("Body cites :", cite_in_body)
print("Refs list  :", cite_in_refs)

missing_in_refs = sorted(set(cite_in_body) - set(cite_in_refs))
unused_refs = sorted(set(cite_in_refs) - set(cite_in_body))
print()
print("Cited in body but missing from references:", missing_in_refs)
print("In references but never cited in body   :", unused_refs)
print()
if not missing_in_refs and not unused_refs:
    print("OK -- references are consistent.")
