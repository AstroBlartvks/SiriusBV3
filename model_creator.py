import random
import torch
import csv

class extract_tensor(torch.nn.Module):
    def forward(self,x):
        tensor, _ = x
        return tensor
    

class ModelCreate(torch.nn.Module):
    def __init__(self, layers):
        super(ModelCreate, self).__init__()
        type_layer = {"FC": torch.nn.Linear, "RNN": torch.nn.RNN, "LSTM": torch.nn.LSTM, "GRU": torch.nn.GRU}
        type_activation = {"Sigmoid": torch.nn.Sigmoid(), "ReLU": torch.nn.ReLU(), "Tanh": torch.nn.Tanh(), "Softmax": torch.nn.Softmax(dim=1)}
        self.layers = list([type_layer[layers[i//2][0]](layers[i//2][1], layers[i//2][2]) if i % 2 == 0 else type_activation[layers[i//2][3]] for i in range(2*len(layers))])
        self.second_layers = []
        for name_id in range(len(self.layers)):
            if "LSTM" in str(self.layers[name_id]) or "RNN" in str(self.layers[name_id]) or "GRU" in str(self.layers[name_id]):
                self.second_layers.append(self.layers[name_id])
                self.second_layers.append(extract_tensor())
            else: self.second_layers.append(self.layers[name_id])
        self.named_layers = list([x[0] for x in layers])
        self.layer_seq = torch.nn.Sequential(*self.second_layers)

    def forward(self, x):
        x = self.layer_seq(x)
        return x
    

class Training:
    def __init__(self, model, input_, target_, basepath, device="cpu"):
        self.model = model
        self.input_size = input_
        self.target_size = target_
        self.basepath = basepath
        self.dataset = []
        self.input_data = []
        self.target_data = []
        self.device = device

    
    def prepare_dataset(self):
        with open(self.basepath) as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar='|')
            self.dataset = list([row for row in reader])
            self.datasize = len(self.dataset)
    

    def train(self, optim, learning_rate, epochs):
        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}[optim](self.model.parameters(), lr=learning_rate)

        test_max = self.datasize // 10 
        valid_max = (2 * self.datasize) // 10 
        train_max = self.datasize - valid_max - test_max

        test_losses = []
        valid_losses = []
        train_losses = []
        
        for epoch_id in range(epochs):
            random.shuffle(self.dataset)
            self.input_data = list([x[:self.input_size] for x in self.dataset])
            self.target_data = list([x[self.input_size:] for x in self.dataset])

            test_loss = 0
            valid_loss = 0
            train_loss = 0
            
            test_input_data = torch.Tensor(self.input_data[:test_max])
            valid_input_data = torch.Tensor(self.input_data[test_max:valid_max])
            train_input_data = torch.Tensor(self.input_data[valid_max:])

            test_target_data = torch.Tensor(self.target_data[:test_max])
            valid_target_data = torch.Tensor(self.target_data[test_max:valid_max])
            train_target_data = torch.Tensor(self.target_data[valid_max:])

            for i in range(train_max):
                optimizer.zero_grad()

                output = self.model(train_input_data[i].unsqueeze(0))
                loss = criterion(output, train_target_data[i].unsqueeze(0))
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

            for i in range(valid_max):
                optimizer.zero_grad()

                output = self.model(valid_input_data[i].unsqueeze(0))
                loss = criterion(output, valid_target_data[i].unsqueeze(0))
                valid_loss += loss.item()
                loss.backward()
                optimizer.step()

            for i in range(test_max):
                output = self.model(test_input_data[i].unsqueeze(0))
                loss = criterion(output, test_target_data[i].unsqueeze(0))
                test_loss += loss.item()

            test_losses.append(test_loss / test_max)
            valid_losses.append(valid_loss / valid_max)
            train_losses.append(train_loss / train_max)

            print(f'Epoch [{epoch_id+1}/{epochs}:', test_losses[-1], valid_losses[-1], train_losses[-1])


