"""A custom exception hierarchy, try/except/else/finally, methods that
mutate self, and cross-object mutation through a registry."""


class BankError(Exception):
    pass


class InsufficientFunds(BankError):
    def __init__(self, needed: int, available: int):
        super().__init__(f"need {needed}, have {available}")
        self.needed = needed
        self.available = available


class Account:
    def __init__(self, owner: str, balance: int = 0):
        self.owner = owner
        self.balance = balance
        self.history: list[str] = []

    def deposit(self, amount: int) -> None:
        if amount <= 0:
            raise ValueError("deposit must be positive")
        self.balance += amount
        self.history.append(f"+{amount}")

    def withdraw(self, amount: int) -> None:
        if amount > self.balance:
            raise InsufficientFunds(amount, self.balance)
        self.balance -= amount
        self.history.append(f"-{amount}")


class Bank:
    def __init__(self):
        self.accounts: dict[str, Account] = {}

    def open(self, owner: str, balance: int = 0) -> Account:
        acct = Account(owner, balance)
        self.accounts[owner] = acct
        return acct

    def transfer(self, src: str, dst: str, amount: int) -> bool:
        try:
            self.accounts[src].withdraw(amount)
            self.accounts[dst].deposit(amount)
        except InsufficientFunds as e:
            print(f"transfer failed: {e} (short by {e.needed - e.available})")
            return False
        except KeyError as e:
            print("no such account:", e)
            return False
        else:
            print(f"moved {amount} {src}->{dst}")
            return True
        finally:
            print("audit:", sorted(self.accounts))


def main() -> None:
    bank = Bank()
    alice = bank.open("alice", 100)
    bank.open("bob")
    alice.deposit(50)
    print(bank.transfer("alice", "bob", 120))
    print(bank.transfer("alice", "bob", 100))
    print(bank.transfer("alice", "carol", 1))
    try:
        alice.deposit(-5)
    except ValueError as e:
        print("rejected:", e)
    for name, acct in sorted(bank.accounts.items()):
        print(name, acct.balance, acct.history)
    print(alice.balance == bank.accounts["alice"].balance)
    print(isinstance(InsufficientFunds(1, 0), BankError), issubclass(InsufficientFunds, Exception))


if __name__ == "__main__":
    main()
