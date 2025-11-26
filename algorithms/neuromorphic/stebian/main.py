from stebpop import StebPop
from popencoder import PopEncoder
import torch

#simple encoding
size = 9  # 3 neurons per symbol

#create population
stebpop = StebPop(size)

# test with correct weights
#stebpop.weights = torch.tensor([[0,1,0],[0,0,1],[1,0,0]], dtype=torch.float32)

#seqence
sequence = [0, 1, 2, 0, 1, 2]
popencoder = PopEncoder(3, size)
#train
iteration = 10
for i in range(iteration):
    print(f"iteration {i}")
    correct = 0
    prev_activation = torch.zeros(size)
    for j in range(len(sequence)):
        input = popencoder.encode(sequence[j])
        prev_activation, pred = stebpop.forward(input, prev_activation)

        # for printing
        output = popencoder.decode(pred)
        if output == sequence[j]:
            correct += 1
    correct_percent = int(100 * correct / len(sequence))
    print(f"correct: {correct_percent}%")

#test
input = popencoder.encode(0)
prev_activation = torch.zeros(size)
for i in range(len(sequence)):
    if i == 0:
        prev_activation, _ = stebpop.forward(input, prev_activation, learning = False)
    else:
        prev_activation, _ = stebpop.forward(None, prev_activation, learning = False)
    output = popencoder.decode(prev_activation)
    print(output)
