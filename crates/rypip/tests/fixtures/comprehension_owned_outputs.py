def main() -> None:
    rows = list([value] for value in range(3))
    rows[0].append(9)
    print(rows)
    print([[value] for value in range(3)])
    print([(value, str(value)) for value in range(3)])
    print([str(row) for row in [[1], [2]]])


if __name__ == "__main__":
    main()
