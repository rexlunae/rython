"""Methods whose unannotated parameters receive class instances.

Python never needs the annotation: `pricer.apply_discount(order)` works
because every caller passes an `Order`. The parameters here carry
instances through a chain of calls (checkout -> add_shipping /
apply_discount -> total) and mutate them, so the printed totals show
each method reading and writing the object it was handed.
"""


class Item:
    def __init__(self, name: str, price: float, qty: int) -> None:
        self.name = name
        self.price = price
        self.qty = qty


class Order:
    def __init__(self, customer: str) -> None:
        self.customer = customer
        self.items: list[Item] = []
        self.discount = 0.0
        self.shipping = 0.0

    def add(self, name: str, price: float, qty: int) -> None:
        self.items.append(Item(name, price, qty))


class Pricer:
    def __init__(self, rate: float, free_over: float) -> None:
        self.rate = rate
        self.free_over = free_over

    def subtotal(self, order):
        total = 0.0
        for item in order.items:
            total += item.price * item.qty
        return total

    def apply_discount(self, order):
        order.discount = round(self.subtotal(order) * self.rate, 2)

    def add_shipping(self, order):
        if self.subtotal(order) - order.discount < self.free_over:
            order.shipping = 4.99

    def total(self, order):
        return round(self.subtotal(order) - order.discount + order.shipping, 2)


class Checkout:
    def __init__(self, pricer: Pricer) -> None:
        self.pricer = pricer
        self.processed = 0

    def run(self, order):
        self.pricer.apply_discount(order)
        self.pricer.add_shipping(order)
        self.processed += 1
        return self.summary(order)

    def summary(self, order):
        return order.customer + ": " + str(self.pricer.total(order))


def main() -> None:
    checkout = Checkout(Pricer(0.1, 50.0))
    small = Order("ana")
    small.add("pen", 1.5, 4)
    big = Order("bo")
    big.add("lamp", 30.0, 2)
    big.add("bulb", 2.25, 4)
    print(checkout.run(small))
    print(checkout.run(big))
    print(small.discount, small.shipping, big.discount, big.shipping)
    print(checkout.processed)


if __name__ == "__main__":
    main()
