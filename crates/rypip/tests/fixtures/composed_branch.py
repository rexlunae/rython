class Inner:
    def __init__(self):
        self.value = 7
class Branch:
    def __init__(self, flag: bool):
        if flag:
            self.inner = Inner()
            print(self.inner.value)
        self.inner = Inner()
        print(self.inner.value)
class Joined:
    def __init__(self, flag: bool):
        if flag:
            self.inner = Inner()
        else:
            self.inner = Inner()
        print(self.inner.value)
Branch(True)
Branch(False)
Joined(True)
Joined(False)
