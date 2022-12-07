import torch


alphabet = [chr(x) for x in range(ord("a"), ord("z")+1)]

# string to int
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

        # optimization
        P = (self.N+1).float()  # model smoothing
        P /= self.N.sum(1, keepdim=True)

        g = torch.Generator().manual_seed(1223123334)

        result = []
        for _ in range(n):
            out = []
            ix = 0
            while True:
                p = P[ix]
                ix = torch.multinomial(
                    p, 1, replacement=True, generator=g).item()
                if itos[ix] == ".":
                    break
                else:
                    out.append("".join(itos[ix]))
                word = "".join(out)
                result.append(word)
            print(f"Predicted Name: {word}")

        loglike = 0
        n = 0
        for w in result:
            chars = ["."] + list(w) + ["."]
            for ch, chi in zip(chars, chars[1:]):
                prob = P[stoi[ch], stoi[chi]]
                logprob = torch.log(prob)
                loglike += logprob
                n += 1

        nll = -loglike
        normll = nll/n
        print(f"NormLogLike: {normll}")
