from core.layer import *
from core.network import *


def train(nn, epochs, learning_step=0.01):
    for epoch in range(epochs):
        if epoch % 100 == 0:
            print(f"\rEpoch {epoch}", end="")
        data = []
        for i in range(-10, 10):
            x = np.array([i * 0.1], dtype=np.float16)
            y = (x + 4) * 4 + 4
            data.append((x, y))
        nn.back_prop(data, learning_step=learning_step)

        if epoch % 100 == 0:
            cost = nn.get_cost(data)
            print(f" - Cost: {cost:.4f}", end="")
        
    print()


def main():
    nn = Network()

    while True:
        cmd = input("> ")
        if not cmd:
            continue
        
        match cmd[0]:
            case "q":
                break

            case "c":
                nn.create(1, [
                    Layer(LayerType.Linear, 1000),
                    Layer(LayerType.Linear, 1),
                ])

            case "t":
                try:
                    epochs, learning_step = cmd[2:].split()
                    epochs = int(epochs)
                    learning_step = float(learning_step)
                except Exception as e:
                    print(e)
                    continue
                train(nn, epochs, learning_step)

            case "s":
                nn.save("models/test.nn")

            case "l":
                nn.load("models/test.nn")

            case _:
                try:
                    data = np.array([float(cmd)], dtype=np.float16)
                except Exception as e:
                    print(e)
                    continue
                
                print(nn.get_result(data))



if __name__ == "__main__":
    main()
