from typing import Optional


class Item:
    def __init__(self, name: str, qty: int, note: Optional[str] = None):
        self.name = name
        self.qty = qty
        self.note = note

    def label(self) -> str:
        if self.note is None:
            return self.name
        return f"{self.name} ({self.note})"


class Perishable(Item):
    def __init__(self, name: str, qty: int, days: int):
        super().__init__(name, qty)
        self.days = days

    def label(self) -> str:
        return super().label() + f" [{self.days}d]"


class Inventory:
    def __init__(self):
        self.items: dict[str, Item] = {}

    def add(self, item: Item) -> None:
        if item.name in self.items:
            self.items[item.name].qty += item.qty
        else:
            self.items[item.name] = item

    def find(self, name: str) -> Optional[Item]:
        return self.items.get(name)

    def take(self, name: str, qty: int) -> int:
        item = self.find(name)
        if item is None:
            raise KeyError(name)
        if qty > item.qty:
            raise ValueError(f"only {item.qty} {name} left")
        item.qty -= qty
        return item.qty

    def report(self) -> list[str]:
        lines = []
        for i, (name, item) in enumerate(sorted(self.items.items())):
            lines.append(f"{i + 1}. {item.label()}: {item.qty}")
        return lines

    def total(self) -> int:
        return sum(item.qty for item in self.items.values())


def main() -> None:
    inv = Inventory()
    inv.add(Item("bolt", 10))
    inv.add(Item("bolt", 5, note="M6"))
    inv.add(Perishable("milk", 2, 7))
    for line in inv.report():
        print(line)
    print(inv.total())
    try:
        inv.take("milk", 5)
    except ValueError as e:
        print("error:", e)
    try:
        inv.take("cheese", 1)
    except KeyError:
        print("no cheese")
    left = inv.take("bolt", 3)
    print(left, inv.find("bolt").label().upper())
    print(inv.total())  # 14 — proves take() mutated the stored Item
    words = [w.strip() for w in "a, b ,c".split(",") if w.strip()]
    print("-".join(words), len(words))


if __name__ == "__main__":
    main()
