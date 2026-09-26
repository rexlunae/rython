# A store deep inside an object passed to a function, and a queue of
# arbitrary values kept in a list[Any] field: both must reach the
# caller's object, as every Python reference does.
import typing


class Engine:
    def __init__(self) -> None:
        self.rpm = 0


class Car:
    def __init__(self, name: str) -> None:
        self.name = name
        self.engine = Engine()


class Garage:
    def __init__(self) -> None:
        self.cars: list[Car] = []
        self.log: list[typing.Any] = []

    def park(self, car: Car) -> None:
        self.cars.append(car)
        self.log.append(car.name)

    def last_logged(self) -> typing.Any:
        if self.log:
            return self.log.pop()
        return None


def rev(car: Car, by: int) -> None:
    car.engine.rpm += by


def main() -> None:
    g = Garage()
    a = Car("a")
    b = Car("b")
    g.park(a)
    g.park(b)
    rev(a, 1000)
    rev(g.cars[1], 250)
    rev(a, 500)
    total = 0
    for c in g.cars:
        total += c.engine.rpm
    print(a.engine.rpm, b.engine.rpm, total)
    print(len(g.log), g.last_logged(), len(g.log))


if __name__ == "__main__":
    main()
