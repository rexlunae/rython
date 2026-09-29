# A method returns a pool built by a manager held in a field that a helper
# method (not __init__) assigns; the caller mutates the returned pool. The
# manager caches pools, so a second request to the same host must see the
# first request's hit: the returned object is the cached one, not a copy.


class Pool:
    def __init__(self, host: str) -> None:
        self.host = host
        self.hits = 0


class Manager:
    def __init__(self) -> None:
        self.pools: dict[str, Pool] = {}
        self.made = 0

    def connection_from_host(self, host: str) -> Pool:
        pool = self.pools.get(host)
        if pool is None:
            self.made += 1
            pool = Pool(host)
            self.pools[host] = pool
        return pool


class Adapter:
    def __init__(self) -> None:
        self.init_manager()

    def init_manager(self) -> None:
        self.manager = Manager()

    def get_connection(self, host):
        conn = self.manager.connection_from_host(host)
        return conn

    def send(self, host: str) -> int:
        conn = self.get_connection(host)
        conn.hits += 1
        return conn.hits


def main() -> None:
    a = Adapter()
    print(a.send("example.com"), a.send("example.com"), a.send("other.org"))
    print(a.manager.made, sorted(a.manager.pools))
    print(a.manager.pools["example.com"].hits)


if __name__ == "__main__":
    main()
