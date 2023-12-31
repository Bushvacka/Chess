import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_utils import AverageMeter
import Chess

LEARNING_RATE = .001
DROPOUT = 0.3
EPOCHS = 10
BATCH_SIZE = 64
NUM_LAYERS = 6
NUM_CHANNELS = 128

MODEL_FOLDER = "models"
MODEL_FILE_NAME = "model.pth.tar"

torch.backends.cudnn.enabled = False

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int = 256, out_channels: int = 256, kernel_size: int = 3, padding: int = 1, activation: nn.Module = nn.ReLU()):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        
        return out

class ResidualBlock(nn.Module):
    def __init__(self, num_channels: int = 256):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(num_channels)
        self.conv2 = ConvBlock(num_channels, activation=nn.Identity())
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = F.relu(out)
        return out

class Model(nn.Module):
    def __init__(self, num_channels: int = 256, num_residuals: int = 15):
        super(Model, self).__init__()

        self.conv = ConvBlock(6, num_channels)

        self.residuals = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_residuals)])

        self.policy_head = nn.Sequential(
            ConvBlock(num_channels, 8, kernel_size=1, padding=0),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, Chess.ACTION_SIZE),
            nn.Softmax(dim=1),
        )

        self.value_head = nn.Sequential(
            ConvBlock(num_channels, 1, kernel_size=1, padding=0),
            nn.Flatten(),
            nn.Linear(8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.to(torch.device('cuda'))

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 6, 8, 8) # Reshape to N X C X H X W

        x = self.conv(x)

        for residual in self.residuals:
            x = residual(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value


    def predict(self, board):
        # Convert to tensor
        board_tensor = torch.FloatTensor(board).contiguous().cuda()
        self.eval()

        # Get prediction
        with torch.no_grad():
            policy, value = self.forward(board_tensor)

        # Bring back to CPU
        policy = policy.contiguous().cpu()
        value = value.contiguous().cpu()

        value = np.reshape(value, (-1)) # Flatten value output

        return policy.data.numpy()[0], value.data.numpy()[0]
    
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
        for _ in tqdm(range(EPOCHS), desc="Epochs"):
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
                canonical_boards = torch.FloatTensor(np.array(canonical_boards)).contiguous().cuda()
                target_policies = torch.FloatTensor(np.array(target_policies)).contiguous().cuda()
                target_values = torch.FloatTensor(np.array(target_values)).contiguous().cuda()

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
    
    def saveCheckpoint(self):
        filepath = os.path.join(MODEL_FOLDER, MODEL_FILE_NAME)
        if not os.path.exists(MODEL_FOLDER):
            print("Making directory {}".format(MODEL_FOLDER))
            os.mkdir(MODEL_FOLDER)
        torch.save({'state_dict': self.state_dict(),}, filepath)

    def loadCheckpoint(self):
        filepath = os.path.join(MODEL_FOLDER, MODEL_FILE_NAME)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath, map_location='cuda')
        self.load_state_dict(checkpoint['state_dict'])