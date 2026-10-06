"""Sets: len, sorted, annotated set returns, membership after mutation."""


def unique_words(text: str) -> set[str]:
    out: set[str] = set()
    for w in text.split():
        out.add(w)
    return out


def letters() -> set[str]:
    return {"b", "a", "c"}


def main() -> None:
    s = unique_words("the cat and the hat")
    print(len(s))
    print(sorted(s))
    s.add("zebra")
    s.discard("cat")
    print(len(s), sorted(s))
    ls = letters()
    ls.add("a")
    print(len(ls), sorted(ls))
    print("a" in ls, "q" in ls)
    nums = {3, 1, 2}
    nums.add(10)
    print(sorted(nums), len(nums))
    print(min(nums), max(nums))
    nums.remove(3)
    try:
        nums.remove(3)
    except KeyError as e:
        print("missing", e)
    try:
        ls.remove("zz")
    except KeyError as e:
        print("missing", e)
    print(sorted(nums), len(nums))


if __name__ == "__main__":
    main()
