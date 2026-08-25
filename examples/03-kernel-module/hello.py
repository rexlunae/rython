"""hello_rython: the classic hello-world Linux kernel module, in Python.

THIS FILE IS THE DRIVER. You maintain the Python; the Makefile next to
it rebuilds hello.ko from this file on every `make` (rypip lowers it to
a #![no_std] Rust crate - C-ABI entry points, .modinfo metadata, printk
lowering - and drives that crate's Kbuild pipeline). The generated Rust
under build/ is an intermediate artifact, like an .o file: read it if
you're curious, but edit here.

Kernel symbols are reached through the rykernel-shim crate, the Rust
compatibility layer that declares what a module may call. No C shim, no
rust-for-linux tree required.

    make            # build build/hello-kmod/hello.ko from this file
    sudo insmod build/hello-kmod/hello.ko
    sudo dmesg | tail -2
    sudo rmmod hello
"""

__module_license__ = "GPL"
__module_author__ = "The rython authors"
__module_description__ = "Hello-world kernel module compiled from Python"
__module_version__ = "0.1.0"

# Kernel resources are explicit imports, resolved at conversion time
# against rykernel-shim's allowlist - a name that is not importable is a
# loud conversion error, never a link failure at insmod time.
from rykernel_shim import ktime_get_real_seconds


def module_init() -> int:
    """Runs in kernel context at insmod time. Return 0 for success."""
    now = ktime_get_real_seconds()
    printk(f"hello_rython: loaded at unix time {now}\n")
    return 0


def module_exit():
    """Runs at rmmod time."""
    printk("hello_rython: goodbye\n")
