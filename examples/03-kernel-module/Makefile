# Build Linux kernel modules FROM THE PYTHON in this directory.
#
# The Python files are the source you maintain. `make` regenerates the
# Rust crates from them on every build (into build/ — a disposable
# intermediate, like .o files) and then drives each generated crate's own
# Kbuild pipeline to produce the .ko. Edit hello.py / driver.py, not the
# generated code.
#
#   make              build hello.ko and rython.ko (needs the prerequisites
#                     printed by the generated Makefiles: linux-headers for
#                     the running kernel, nightly rust + the
#                     x86_64-unknown-none target, GNU ld >= 2.36)
#   make check        host-only verification, no kernel headers needed:
#                     regenerate both crates and cargo-check them
#   make tool         build the userspace driver binary from driver.py
#                     (the same Python also declares the device manifest)
#   make load         insmod both modules       (sudo)
#   make unload       rmmod both modules        (sudo)
#   make dmesg        show the modules' latest kernel log lines
#   make clean        drop build/
#
# One-time toolchain setup for the .ko build:
#   rustup toolchain install nightly --profile minimal
#   rustup +nightly target add x86_64-unknown-none
#   rustup +nightly component add rust-src

# rypip from this checkout; override RYPIP to use an installed binary.
RYPIP ?= cargo run -q -p rypip --manifest-path ../../Cargo.toml --
BUILD := build

.PHONY: all check tool load unload dmesg clean \
        convert-hello convert-rython convert-tool

all: $(BUILD)/hello-kmod/hello.ko $(BUILD)/rython-kmod/rython.ko

# --- regenerate the crates from the Python (always: the Python is the
# --- source of truth, conversion is cheap, and stale output is worse) ---

convert-hello: hello.py
	$(RYPIP) convert hello.py --out $(BUILD)/hello-kmod --kernel-module

convert-rython: driver.py
	$(RYPIP) convert driver.py --out $(BUILD)/rython-kmod --kernel-module

convert-tool: driver.py
	$(RYPIP) convert driver.py --out $(BUILD)/rython-tool --driver

# --- kernel modules: each generated crate carries its own Kbuild
# --- Makefile (cargo -Zbuild-std staticlib -> ld -r -> Kbuild/modpost) ---

$(BUILD)/hello-kmod/hello.ko: convert-hello
	$(MAKE) -C $(BUILD)/hello-kmod

$(BUILD)/rython-kmod/rython.ko: convert-rython
	$(MAKE) -C $(BUILD)/rython-kmod

# --- host-only verification (CI-friendly: no headers, no nightly) ---

check: convert-hello convert-rython convert-tool
	cargo check -q --manifest-path $(BUILD)/hello-kmod/Cargo.toml
	cargo check -q --manifest-path $(BUILD)/rython-kmod/Cargo.toml
	cargo test -q --manifest-path $(BUILD)/rython-tool/Cargo.toml
	@echo "check: OK (crates regenerate from the Python and pass on the host)"

# --- the userspace half of the rython device ---

tool: convert-tool
	cargo build -q --release --manifest-path $(BUILD)/rython-tool/Cargo.toml
	@echo "built $(BUILD)/rython-tool/target/release/driver"

# --- load / unload / logs ---

load: all
	sudo insmod $(BUILD)/hello-kmod/hello.ko
	sudo insmod $(BUILD)/rython-kmod/rython.ko
	@sudo dmesg | tail -4

unload:
	sudo rmmod rython hello
	@sudo dmesg | tail -4

dmesg:
	sudo dmesg | grep -E "hello_rython|rython" | tail -10

clean:
	rm -rf $(BUILD)
