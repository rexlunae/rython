"""A tiny register-file device: kernel module + userspace driver, both
maintained as this one Python file.

THIS FILE IS THE DRIVER. You edit the Python; `make` (see the Makefile
next to this file) rebuilds both halves from it every time - the
generated Rust under build/ is an intermediate artifact, never the
thing you maintain:

  make          -> build/rython-kmod/rython.ko
      The device manifest below (__device_name__, __bufsz__, ...)
      becomes a misc character device (/dev/rython0): a pure-Rust .ko
      with full file_operations, an ioctl ABI, and .modinfo metadata.

  make tool     -> the userspace driver binary
      The classes and functions below - ordinary Python, compiled by
      the full transpiler - become the driver logic, wrapped in
      generated open/read/write/ioctl syscall glue that talks to that
      device node (the UIO pattern: the kernel side stays a dumb, safe
      byte ring; the smarts live in user space, in Python).

Workflow: edit here -> `python3 driver.py` (CPython smoke test) ->
`make load` -> talk to /dev/rython0 with the tool -> `make unload`.
"""

__module_name__ = "rython"
__module_license__ = "GPL"
__module_author__ = "The rython authors"
__module_description__ = "byte-ring misc device with rython-compiled driver logic"
__module_version__ = "0.1.0"

# The device manifest - the shared ABI of the kernel and userspace halves.
__device_path__ = "/dev/rython0"
__device_name__ = "rython0"
__bufsz__ = 4096
__magic__ = 0x52594854  # "RYHT"
__device_mode__ = 0o600
__ioc_reset__ = 0x5201  # _IO(0x52, 1)
__ioc_stats__ = 0x80285202  # _IOR(0x52, 2, 40-byte stats struct)


def parse_hex(s: str) -> int:
    """Parse a hex string ("2a", "DEADBEEF") to an int; -1 if invalid."""
    digits = "0123456789abcdef"
    if len(s) == 0:
        return -1
    lowered = s.lower()
    value = 0
    i = 0
    while i < len(lowered):
        idx = digits.find(lowered[i])
        if idx < 0:
            return -1
        value = value * 16 + idx
        i = i + 1
    return value


def crc8(data: bytes) -> int:
    """CRC-8 (poly 0x07, init 0, no reflection) over the device echo."""
    crc = 0
    i = 0
    while i < len(data):
        crc = crc ^ int(data[i])
        bit = 0
        while bit < 8:
            if crc & 1:
                crc = (crc >> 1) ^ 0x07
            else:
                crc = crc >> 1
            bit = bit + 1
        i = i + 1
    return crc


def within(value, low, high):
    """Inclusive range check. No annotations: rython infers a generic,
    comparison-bounded signature from how the parameters are used."""
    return low <= value and value <= high


class RegisterBank:
    """Base class: bounds-checked byte-register storage."""

    def __init__(self, regs: dict[int, int], size: int):
        self.regs = regs
        self.size = size

    def load(self, addr: int) -> int:
        return self.regs.get(addr, 0)

    def store(self, addr: int, value: int) -> bool:
        if not within(addr, 0, self.size - 1):
            return False
        if not within(value, 0, 255):
            return False
        self.regs[addr] = value
        return True

    def clear(self) -> None:
        self.regs.clear()

    def dump(self) -> str:
        entries = []
        for addr in sorted(self.regs.keys()):
            value = self.regs[addr]
            entries.append(f"{addr}:{value}")
        return " ".join(entries)


class Device(RegisterBank):
    """The driver protocol, layered on the register bank by inheritance."""

    def __init__(self, regs: dict[int, int], name: str):
        super().__init__(regs, 8)
        self.name = name
        self.ops = 0
        self.reads = 0

    def clear(self) -> None:
        # Extend, don't replace: the base clears storage, we reset stats.
        super().clear()
        self.ops = 0
        self.reads = 0

    def handle(self, line: str) -> str:
        parts = line.split()
        if len(parts) == 0:
            return "ERR empty"
        cmd = parts[0]
        if cmd == "WRITE":
            if len(parts) != 3:
                return "ERR bad write"
            addr = parse_hex(parts[1])
            value = parse_hex(parts[2])
            if addr < 0 or value < 0:
                return "ERR bad write"
            if not self.store(addr, value):
                return "ERR bad write"
            self.ops = self.ops + 1
            return f"OK {value}"
        if cmd == "READ":
            if len(parts) != 2:
                return "ERR bad read"
            addr = parse_hex(parts[1])
            if addr < 0:
                return "ERR bad read"
            self.reads = self.reads + 1
            value = self.load(addr)
            return f"VAL {value}"
        if cmd == "DUMP":
            return self.dump()
        if cmd == "STATS":
            return f"ops={self.ops} reads={self.reads}"
        if cmd == "RESET":
            self.clear()
            return "OK reset"
        if cmd == "HELP":
            return "READ <hex> | WRITE <hex> <hex> | DUMP | STATS | RESET | HELP"
        return f"ERR unknown cmd: {cmd}"


if __name__ == "__main__":
    # CPython smoke test of the protocol - the same assertions the
    # generated driver crate runs as Rust unit tests.
    dev = Device({}, "selftest")
    print(dev.handle("WRITE 2 2a"))
    print(dev.handle("READ 2"))
    print(dev.handle("DUMP"))
    print(dev.handle("STATS"))
    print(dev.handle("RESET"))
    print(dev.handle("NOPE"))
    print(crc8("WRITE 2 2a\n".encode()))
    print(parse_hex("DEADBEEF"))
