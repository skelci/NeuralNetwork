from core.layer import *
from core.network import *



def main():
    nn = Network()
    nn.create(2, [
        Layer(3, LayerType.ELU),
        Layer(1, LayerType.Sigmoid)
    ])
    # nn.load("models/test.nn")

    a, b = np.int16(input("Enter two numbers: ").split())
    print(nn.get_output(np.array([a, b]))[0])

    nn.save("models/test.nn")


if __name__ == "__main__":
    main()
