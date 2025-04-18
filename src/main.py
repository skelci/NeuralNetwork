from core.layer import *
from core.network import *



def main():
    nn = Network()
    nn.create(1, [
        Layer(1, LayerType.Raw),
    ])
    # nn.load("models/test.nn")

    for epoch in range(1000):
        if epoch % 100 == 0:
            print(f"\rEpoch {epoch}", end="", flush=True)
        data = []
        for _ in range(100):
            x = np.random.randint(-10, 10, size=(1,))
            y = x * 2 - 10 + np.random.rand(1) * 10 - 5
            data.append((x, y))
        nn.back_prop(data, 1e-2)

    print()

    while True:
        a = input("Enter number: ")
        if a == "q":
            break
        a = np.float16(a)
        print(nn.get_result(np.array([a])))

    nn.save("models/test.nn")


if __name__ == "__main__":
    main()
