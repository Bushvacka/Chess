import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_utils import AverageMeter
from torchviz import make_dot
import Chess


LEARNING_RATE = .001
DROPOUT = 0.3
EPOCHS = 10
BATCH_SIZE = 64
NUM_CHANNELS = 512
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.num_channels = NUM_CHANNELS
        self.board_size = Chess.BOARD_SIZE
        self.dropout = DROPOUT

        self.conv = nn.Sequential(
            nn.Conv2d(1, self.num_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_channels),
            nn.ReLU(),
            nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_channels),
            nn.ReLU(),
            nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1),
            nn.BatchNorm2d(self.num_channels),
            nn.ReLU(),
            nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1),
            nn.BatchNorm2d(self.num_channels),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.num_channels * (self.board_size - 4) * (self.board_size - 4), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.fc_policy = nn.Linear(512, Chess.ACTION_SIZE)
        self.fc_value = nn.Linear(512, 1)

    def forward(self, state):
        # s: batch_size x board_x x board_y
        state = state.view(-1, 1, self.board_size, self.board_size)  # batch_size x 1 x board_x x board_y
        state = self.conv(state)  # batch_size x num_channels x (board_x-4) x (board_y-4)
        state = state.view(-1, self.num_channels * (self.board_size - 4) * (self.board_size - 4))
        state = self.fc(state)  # batch_size x 512

        pi = self.fc_policy(state)  # batch_size x action_size
        v = self.fc_value(state)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)


    def train_nn(self, examples):
        """
        Train the model on the set of examples EPOCH times. Process the examples in batches
        of size BATCH_SIZE, updating the parameters of the model after every batch.
        Parameters:
            examples: Training examples of the form (canonicalBoard, policy, value)
        """
        optimizer = optim.Adam(self.parameters(), lr = LEARNING_RATE)
        policy_losses = AverageMeter("Policy Loss", 10)
        value_losses = AverageMeter("Value Loss", 10)
        num_examples = len(examples)
        batch_count = int(num_examples / BATCH_SIZE)

        self.train()
        for _ in range(EPOCHS):
            for batch in range(batch_count):
                # Extract a batch of examples
                start = batch * BATCH_SIZE
                stop = start + BATCH_SIZE
                if stop > num_examples:
                    batch_examples = examples[num_examples - BATCH_SIZE:num_examples]
                else:
                    batch_examples = examples[start:stop]

                # Get the boards and corresponding target policies and values
                canonical_boards = [example[0] for example in batch_examples]
                target_policies = [example[1] for example in batch_examples]
                target_values = [example[2] for example in batch_examples]

                # Convert to tensors
                canonical_boards = torch.FloatTensor(np.array(canonical_boards).astype(np.float64))
                target_policies = torch.FloatTensor(np.array(target_policies))
                target_values = torch.FloatTensor(np.array(target_values).astype(np.float64))

                # Get model prediction
                policies, values = self.forward(canonical_boards)

                # Calculate the losses
                policy_loss = -torch.sum(target_policies * policies) / target_policies.size()[0]
                value_loss = torch.sum((target_values - values.view(-1)) ** 2) / target_values.size()[0]
                loss = policy_loss + value_loss

                # Save losses
                policy_losses.update(policy_loss.item())
                value_losses.update(value_loss.item())

                # Gradient Descent
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                

    def predict(self, board):
        board_tensor = torch.FloatTensor(board)
        board_tensor = board_tensor.view(1, 8, 8)
        self.eval()
        with torch.no_grad():
            policy, value = self.forward(board_tensor)
        policy = torch.exp(policy) # Normalize policy vectors
        value = np.reshape(value, (-1))
        return policy.data.numpy()[0], value.data.numpy()[0]
    
    
    def saveCheckpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Making directory {}".format(folder))
            os.mkdir(folder)
        torch.save({'state_dict': self.state_dict(),}, filepath)

    def loadCheckpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])

    def visualizeModel(self):
        board = Chess.getInitBoard()
        board_tensor = torch.FloatTensor(Chess.integerRepresentation(board))
        board_tensor = board_tensor.view(1, self.board_size, self.board_size)
        self.eval()
        out = self.forward(board_tensor)
        make_dot(out, params=dict(self.named_parameters())).render(filename="visualization", format='png')
