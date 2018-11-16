
def source_read(path):
    content = ""
    with open(path) as f:
        content = f.read()
    return content
