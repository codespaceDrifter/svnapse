import torch

class PopEncoder:
    def __init__(self, dictionary_size, population_size):
        assert population_size >= dictionary_size, "dictionary size must be greater than population size"
        assert population_size % dictionary_size == 0, "population size must be divisible by dictionary size"
        self.population_size = population_size
        self.dictionary_size = dictionary_size
        self.dict_to_pop_scale = population_size / dictionary_size

    def encode(self, input: int):
        input_vector = torch.zeros(self.population_size)
        start = int(input * self.dict_to_pop_scale)
        end = int(start + self.dict_to_pop_scale)
        input_vector[start:end] = 1
        return input_vector

    # find the token "bracket" with the most activation. if ties, pick a random one. 
    def decode(self, activation):
        sums = torch.zeros(self.dictionary_size)
        for i in range(self.dictionary_size):
            start = int(i * self.dict_to_pop_scale)
            end = int(start + self.dict_to_pop_scale)
            sums[i] = activation[start:end].sum()
    
        max_index = sums.argmax().item()
        return max_index
            