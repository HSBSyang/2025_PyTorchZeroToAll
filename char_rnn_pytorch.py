import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from text_loader import TextDataset, collate_fn
from torch.nn.utils.rnn import pad_sequence

# Default values
hidden_size = 100
n_layers = 3
batch_size = 1
n_epochs = 100
n_characters = 128  # ASCII
learning_rate = 0.001


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embed = self.embedding(input) # B x S x I (embedding size)
        print(f"embed size: {embed.size()}")
        output, hidden = self.gru(embed, hidden)
        output = self.linear(output) # B x S x O
        return output, hidden

    def init_hidden(self, batch_size=1):
        if torch.cuda.is_available():
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda()
        else:
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)

        return hidden


def generate(decoder, prime_str='A', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden(batch_size=1)
    prime_input = torch.tensor([[ord(c) for c in prime_str]])
    if torch.cuda.is_available():
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    if prime_input.size(1) > 1:
        _, hidden = decoder(prime_input[:, :-1], hidden)

    inp = prime_input[:, -1:]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = chr(top_i)
        predicted += predicted_char
        inp = torch.tensor([[ord(predicted_char)]])
        if torch.cuda.is_available():
            inp = inp.cuda()

    return predicted

# Train for a given src and target
# It feeds single string to demonstrate seq2seq
# It's extremely slow, and we need to use (1) batch and (2) data parallelism
# http://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html.


def train(decoder, criterion, decoder_optimizer, src, target):
    hidden = decoder.init_hidden(batch_size=src.size(0))
    decoder.zero_grad()
    loss = 0

    output, hidden = decoder(src, hidden)

    # The output of the RNN is B x S x O, and the target is B x S.
    # The CrossEntropyLoss expects the output to be B x O x S, so we need to permute it.
    loss = criterion(output.permute(0, 2, 1), target)

    loss.backward()
    decoder_optimizer.step()

    return loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=100, help='size of RNN hidden state')
    parser.add_argument('--n_layers', type=int, default=3, help='number of layers in the RNN')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()

    hidden_size = args.hidden_size
    n_layers = args.n_layers
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate

    decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
    if torch.cuda.is_available():
        decoder.cuda()

    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(dataset=TextDataset(),
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)

    print("Training for %d epochs..." % n_epochs)
    for epoch in range(1, n_epochs + 1):
        for i, (src, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                src = src.cuda()
                target = target.cuda()
            loss = train(decoder, criterion, decoder_optimizer, src, target)

            if i % 100 == 0:
                print('[(%d %d%%) loss: %.4f]' %
                      (epoch, epoch / n_epochs * 100, loss))
                print(generate(decoder, 'Wh', 100), '\n')
