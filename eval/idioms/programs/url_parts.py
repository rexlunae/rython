from urllib.parse import urlparse, urlsplit


def describe(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname
    if host is None:
        return "no host in " + url
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return f"{parsed.scheme} {host} {port} {parsed.path!r}"


def credentials(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.username} {parts.password}"


def main() -> None:
    for url in ["https://Example.COM/a/b", "http://h:8080/x?q=1", "file:///tmp/x", "http://[::1]:8/"]:
        print(describe(url))
    print(credentials("http://alice:secret@example.com/"))
    print(credentials("http://example.com/"))
    try:
        print(describe("http://h:abc/"))
    except ValueError as e:
        print("ValueError:", e)


if __name__ == "__main__":
    main()
