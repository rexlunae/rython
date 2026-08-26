import hashlib

def get_md5(*args, **kwargs) -> str:
    m = hashlib.md5()
    return m.hexdigest()

def main() -> None:
    print(get_md5())

if __name__ == "__main__":
    main()
