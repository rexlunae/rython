"""dict of lists, tuple sorting, list.remove/index, del of a dict key,
nested loops with membership tests, sets."""


def add_slot(cal: dict[str, list[tuple[int, str]]], day: str, hour: int, what: str) -> None:
    cal.setdefault(day, []).append((hour, what))


def busiest(cal: dict[str, list[tuple[int, str]]]) -> str:
    best = ""
    best_n = -1
    for day in sorted(cal):
        if len(cal[day]) > best_n:
            best, best_n = day, len(cal[day])
    return best


def main() -> None:
    cal: dict[str, list[tuple[int, str]]] = {}
    add_slot(cal, "mon", 9, "standup")
    add_slot(cal, "mon", 14, "review")
    add_slot(cal, "tue", 11, "1:1")
    add_slot(cal, "mon", 10, "planning")
    add_slot(cal, "wed", 16, "retro")
    for day in sorted(cal):
        slots = sorted(cal[day])
        print(day, [f"{h:02d}:00 {w}" for h, w in slots])
    print("busiest:", busiest(cal))
    cal["mon"].remove((14, "review"))
    print(len(cal["mon"]), cal["mon"].index((10, "planning")))
    del cal["wed"]
    print(sorted(cal), "wed" in cal)
    hours = {h for slots in cal.values() for h, _ in slots}
    print(sorted(hours), 9 in hours, len(hours))
    free = [h for h in range(9, 13) if h not in hours]
    print(free)
    names = set()
    for slots in cal.values():
        for _, w in slots:
            names.add(w)
    print(sorted(names), sorted(names & {"standup", "lunch"}))


if __name__ == "__main__":
    main()
