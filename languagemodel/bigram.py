import torch

# string to int
alphabet = [chr(x) for x in range(ord("a"), ord("z")+1)]
stoi = {ch: i+1 for i, ch in enumerate(alphabet)}
stoi["."] = 0

# int to string
itos = {i+1: ch for i, ch in enumerate(alphabet)}
itos[0] = "."


class Bigram():
    def __init__(self, words):
        '''
        words -> List of words to train. ["word", "word"]
        '''
        self.N = []

        # array for all the alphabets + "."
        self.N = torch.zeros(27, 27, dtype=torch.int32)

        for w in words[:]:
            chars = ["."] + list(w) + ["."]
            for c, ci in zip(chars, chars[1:]):
                self.N[stoi[c], stoi[ci]] += 1

    def predict(self, n):
        '''
        n -> No of predicted words ["word","word",.."n"]
        '''
        g = torch.Generator().manual_seed(1223123334)
        result = []
        for _ in range(n):
            out = []
            ix = 0
            while True:
                p = self.N[ix].float()
                p = p / sum(p)
                ix = torch.multinomial(
                    p, 1, replacement=True, generator=g).item()
                if itos[ix] == ".":
                    break
                else:
                    out.append("".join(itos[ix]))
            result.append("".join(out))
        return result
