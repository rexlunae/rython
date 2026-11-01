def main():
    big = 18446744073709551616
    print(big)
    total = big + 7
    total += big
    print(total)
    print(big - 3)
    print(3 - big)
    print(big * 3)
    print(-big)
    print(big // -3)
    print(big % -3)
    print(big > 9223372036854775807)
    print(big == big + 1)
    print(hash(big))
    print(bool(big), not (big - big))
    small = big - big
    print(small + 9223372036854775807 + 1)
    other = 0
    other += big
    print(other)
    other = 0
    print(other)
    alias = big
    print(alias, big)
    print(str(big), repr(big), abs(-big), int(big))
    print(f"exact={big} repr={big!r}")
    print(~big, +big)
    rounded = float(big)
    print(big == rounded, big + 1 == rounded, big + 1 > rounded, rounded < big + 1)
    print(big == True, big > 0, big < float("inf"), big > float("-inf"), big < float("nan"))
    try:
        print(big // 0)
    except ZeroDivisionError:
        print("zero //")
    try:
        print(big % 0)
    except ZeroDivisionError:
        print("zero %")


if __name__ == "__main__":
    main()
