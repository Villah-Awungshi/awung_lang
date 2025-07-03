import sys
from awunglang import AwungLang  # This imports your interpreter

def main():
    if len(sys.argv) < 2:
        print("Usage: python awunglang_cli.py <source-file.awung>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        sys.exit(1)

    interpreter = AwungLang()
    output = interpreter.run(source_code)
    for line in output:
        print(line)

if __name__ == "__main__":
    main()
