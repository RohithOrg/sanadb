"""Render python/sanadb.pyi as a Markdown API reference (docs/python/api.md).

    python scripts/pyi_to_md.py > docs/python/api.md
"""
import ast
import sys

src = open("python/sanadb.pyi").read()
tree = ast.parse(src)
lines = src.splitlines()

def sig(fn):
    a = fn.args
    parts = []
    pos = a.posonlyargs + a.args
    defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    for arg, d in zip(pos, defaults):
        if arg.arg == "self":
            continue
        s = arg.arg + (": " + ast.unparse(arg.annotation) if arg.annotation else "")
        if d is not None:
            s += " = " + ast.unparse(d)
        parts.append(s)
    if a.kwonlyargs:
        parts.append("*")
        for arg, d in zip(a.kwonlyargs, a.kw_defaults):
            s = arg.arg + (": " + ast.unparse(arg.annotation) if arg.annotation else "")
            if d is not None:
                s += " = " + ast.unparse(d)
            parts.append(s)
    ret = " -> " + ast.unparse(fn.returns) if fn.returns else ""
    return f"{fn.name}({', '.join(parts)}){ret}"

out = ["# Python API", "", "Generated from the package's type stubs (`python/sanadb.pyi`), which ship in the wheel, so an editor shows the same signatures and docstrings.", "", "```python", "import sanadb", "```", ""]
for node in tree.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
        doc = ""
        out.append(f"## `{name}`")
        out.append("")
        out.append("```python")
        out.append(f"{name} = {ast.unparse(node.value)}")
        out.append("```")
        out.append("")
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) and out and out[-2].startswith("```"):
        out.insert(len(out) - 1, node.value.value.strip())
        out.insert(len(out) - 1, "")
    if isinstance(node, ast.ClassDef):
        out.append(f"## class `{node.name}`")
        out.append("")
        if ast.get_docstring(node):
            out.append(ast.get_docstring(node))
            out.append("")
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                decorators = [ast.unparse(d) for d in item.decorator_list]
                if "property" in decorators:
                    out.append(f"### `{item.name}` (property)")
                else:
                    out.append(f"### `{sig(item)}`")
                out.append("")
                d = ast.get_docstring(item)
                if d:
                    out.append(d)
                    out.append("")
print("\n".join(out))
