from typing import Optional


class Account:
    def __init__(self, owner: str, balance: int):
        self.owner = owner
        self.balance = balance
        self.history: list[int] = []

    def deposit(self, amount: int) -> None:
        self.balance += amount
        self.history.append(amount)

    def withdraw(self, amount: int) -> bool:
        if amount > self.balance:
            return False
        self.balance -= amount
        self.history.append(-amount)
        return True

    def describe(self) -> str:
        return f"{self.owner}: {self.balance} ({len(self.history)} moves)"


class Bank:
    def __init__(self):
        self.accounts: dict[str, Account] = {}
        self.audit: list[Account] = []

    def open(self, owner: str, balance: int) -> Account:
        acct = Account(owner, balance)
        self.accounts[owner] = acct
        self.audit.append(acct)
        return acct

    def find(self, owner: str) -> Optional[Account]:
        return self.accounts.get(owner)

    def transfer(self, src: str, dst: str, amount: int) -> bool:
        a = self.find(src)
        b = self.find(dst)
        if a is None or b is None:
            return False
        if not a.withdraw(amount):
            return False
        b.deposit(amount)
        return True

    def total(self) -> int:
        return sum(a.balance for a in self.accounts.values())


class Book:
    def __init__(self):
        self.entries: list[int] = []

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, n: int) -> None:
        self.entries.append(n)


class Journal(Book):
    pass


def has_entries(b: Book) -> str:
    # Truth through the root's sum type: an empty book is False.
    if bool(b):
        return "yes"
    return "no"


def main() -> None:
    bank = Bank()
    alice = bank.open("alice", 100)
    bank.open("bob", 20)
    alice.deposit(5)
    print(bank.transfer("alice", "bob", 30))
    print(bank.transfer("bob", "alice", 500))
    print(bank.transfer("carol", "bob", 1))
    for acct in bank.audit:
        print(acct.describe())
    first = bank.audit[0]
    first.balance = 1
    print(bank.find("alice").describe())
    print(bank.total())
    # Identity and the default `==` (identity) on shared objects: the
    # fetched local IS the stored object; a distinct account is neither.
    other = Account("alice", 1)
    print(first is alice, first == alice, first is other, first == other)
    # Truth: a plain object is True; a book's truth is its length,
    # inherited by the journal, through the sum type too.
    shelf: list[Book] = [Book(), Journal()]
    print(bool(first), bool(shelf[0]), has_entries(shelf[1]))
    shelf[1].add(3)
    j = shelf[1]
    print(bool(j), len(j.entries), has_entries(shelf[1]))


if __name__ == "__main__":
    main()
