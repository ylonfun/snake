import ast, glob, os

os.chdir(r"D:\Duobao-workpalce\NN-net-learn\snake-main")
for f in sorted(glob.glob("*.py")):
    src = open(f, encoding="utf-8").read()
    tree = ast.parse(src)
    imported = {}   # 绑定名 -> 来源
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imported[n.asname or n.name.split(".")[0]] = n.name
        elif isinstance(node, ast.ImportFrom):
            for n in node.names:
                if n.name == "*":
                    continue
                imported[n.asname or n.name] = (node.module or "") + "." + n.name
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            used.add(node.value.id)
    unused = [name for name in imported if name not in used]
    if unused:
        print(f, "未使用:", ", ".join(unused))
