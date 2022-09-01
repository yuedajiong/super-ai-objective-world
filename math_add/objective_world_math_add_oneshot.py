class TpjWorld(torch.utils.data.Dataset):
    def __init__(self, split, ndigit=(2,3)[0]):
        self.split = split
        self.ndigit = ndigit
        number_total = (10**self.ndigit)**2
        number_valid = int(number_total*0.2)
        generator = torch.Generator()
        generator.manual_seed(122333)  #fix
        permutation = torch.randperm(number_total, generator=generator)
        self.indexes = permutation[:number_valid] if split=='valid' else permutation[number_valid:]

    def get_vocab_size(self):
        return 10  #digits 0..9

    def get_block_size(self):
        return 3*self.ndigit+1-1  #a,b,a+b, and +1 due to potential carry overflow, but then also -1 because very last digit doesn't ever plug back as there is no explicit <EOS> token to predict, it is implied

    def __len__(self):
        return self.indexes.nelement()

    def __getitem__(self, index):
        index = self.indexes[index].item()
        nd = 10**self.ndigit
        a = index // nd
        b = index %  nd
        c = a + b
        astr = f'%0{self.ndigit}d' % a
        bstr = f'%0{self.ndigit}d' % b
        cstr = (f'%0{self.ndigit+1}d' % c)[::-1]  #reverse c to make addition easier, +1 means carry-overflow
        render = astr + bstr + cstr
        encoded = [int(s) for s in render]  #convert each character to its token index
        x = torch.tensor(encoded[:-1], dtype=torch.long)  #x is input to GPT
        #x[self.ndigit*2:] = 10  #如果不注释此行，相当于两个2加数，和部分加和的信息，所以学习很快； 如果放开get_vocab_size：10+1
        y = torch.tensor(encoded[1:], dtype=torch.long)   #y is the associated expected outputs, predict the next token in the sequence
        y[:self.ndigit*2-1] = -1  #only train in the output locations. -1 will mask loss to zero：cross_entropy(..., ignore_index=-1)
        return x, y
