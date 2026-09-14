"""Complex component locals in mixed tuple returns (#366).

The real/imaginary COMPONENT of a complex is a float; a function that
returns such a component local inside a tuple that also mixes complex
arithmetic produces a typed (Complex, f64, ...) tuple, not a collapse.
State is observable in the output: each value prints.
"""


def decompose():
    a = 1j
    s = a + 2j            # Complex
    r = abs(3j)           # f64
    re = (3 + 4j).real    # f64
    im = (3 + 4j).imag    # f64
    return s, r, re, im


def full():
    a = 1j
    b = 2j
    s = a + b             # Complex
    m = a * b             # Complex
    d = a / b             # Complex
    r = abs(3j)           # f64
    c = (1 + 2j).conjugate()  # Complex
    re = (3 + 4j).real    # f64
    im = (3 + 4j).imag    # f64
    return s, m, d, r, c, re, im


def via_local():
    z = 2j
    re = z.real
    im = z.imag
    c = z.conjugate()
    return c, re, im


def main() -> None:
    s, r, re, im = decompose()
    print(s, r, re, im, sep="|")
    s2, m2, d2, r2, c2, re_, im_ = full()
    print(s2, m2, d2, r2, c2, re_, im_, sep="|")
    c, re3, im3 = via_local()
    print(c, re3, im3, sep="|")


if __name__ == "__main__":
    main()
