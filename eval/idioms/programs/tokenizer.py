"""A while-loop scanner over a string with slicing and char predicates,
a list of tuples, and a stack-based evaluator with pop/append."""


def tokenize(src: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    i = 0
    while i < len(src):
        ch = src[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit():
            j = i
            while j < len(src) and src[j].isdigit():
                j += 1
            tokens.append(("num", src[i:j]))
            i = j
        elif ch.isalpha():
            j = i
            while j < len(src) and (src[j].isalnum() or src[j] == "_"):
                j += 1
            tokens.append(("name", src[i:j]))
            i = j
        elif ch in "+-*/()":
            tokens.append(("op", ch))
            i += 1
        else:
            tokens.append(("bad", ch))
            break
    return tokens


def rpn(tokens: list[str]) -> int:
    stack: list[int] = []
    for t in tokens:
        if t.isdigit():
            stack.append(int(t))
        else:
            b = stack.pop()
            a = stack.pop()
            if t == "+":
                stack.append(a + b)
            elif t == "-":
                stack.append(a - b)
            elif t == "*":
                stack.append(a * b)
            else:
                stack.append(a // b)
    return stack[0]


def main() -> None:
    toks = tokenize("x1 + 42 * (y_2 - 7)")
    print(len(toks))
    for kind, text in toks:
        print(f"{kind}:{text}", end=" ")
    print()
    kinds = [k for k, _ in toks]
    print(kinds.count("name"), kinds.index("op"), "bad" in kinds)
    print(tokenize("3 $ 4"))
    print(rpn("3 4 + 2 *".split()), rpn("20 4 / 3 -".split()))
    src = "hello world"
    print(src[:5].upper(), src[-5:], src[::2], src[::-1])


if __name__ == "__main__":
    main()
