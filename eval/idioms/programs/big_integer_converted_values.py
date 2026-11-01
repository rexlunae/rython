def main():
    big = 18446744073709551616
    values = [str(big), repr(big + 1)]
    print(values)
    print((str(big),))
    mapping = {str(big): repr(big + 1)}
    print(mapping[str(big)])
    mapping[str(big)] = str(big + 2)
    print(mapping[str(big)])
    print(str(big)[0])
    print(values[len(str(big)) - 20])
    text = ""
    for char in str(big):
        text += char
    print(text)
    print([str(big + i) for i in range(2)])
    print(sorted({str(big), repr(big + 1)}))
    try:
        raise ValueError(str(big))
    except ValueError as error:
        print(str(error))


if __name__ == "__main__":
    main()
