import torch
import torchvision


class LinearModel(torch.nn.Module):
    def __init__(self, hyperparameters: dict):
        super(LinearModel, self).__init__()

        # Get model config
        self.input_dim = hyperparameters['input_dim']
        self.output_dim = hyperparameters['output_dim']
        self.hidden_dims = hyperparameters['hidden_dims']
        self.negative_slope = hyperparameters.get("negative_slope", .2)

        # Create layer list
        self.layers = torch.nn.ModuleList([])
        all_dims = [self.input_dim, *self.hidden_dims, self.output_dim]
        for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
            self.layers.append(torch.nn.Linear(in_dim, out_dim))

        self.num_layers = len(self.layers)

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
            x = torch.nn.functional.leaky_relu(
                x, negative_slope=self.negative_slope)
        x = self.layers[-1](x)
        return torch.nn.functional.softmax(x, dim=-1)



class CNNModel(torch.nn.Module):
    def __init__(self, hyperparameters: dict):
        super(CNNModel, self).__init__()

        self.num_filters = hyperparameters["num_filters"]
        self.kernel_size = hyperparameters["kernel_size"]
        self.stride = hyperparameters["stride"]
        self.dropout_rate = hyperparameters["dropout_rate"]
        self.num_layers = hyperparameters["num_layers"]
    
    # 첫 번째 컨볼루션 레이어
        layers = [
            torch.nn.Conv2d(1, self.num_filters[0], kernel_size=self.kernel_size, stride=self.stride, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        ]
        
        # 두 번째 컨볼루션 레이어
        layers.append(torch.nn.Conv2d(self.num_filters[0], self.num_filters[1], kernel_size=self.kernel_size, stride=self.stride, padding=1))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool2d(2))
        
        # 추가적인 컨볼루션 레이어들 (num_layers에 따라)
        if self.num_layers > 2:
            layers.append(torch.nn.Conv2d(self.num_filters[1], self.num_filters[1], kernel_size=self.kernel_size, stride=self.stride, padding=1))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.MaxPool2d(2))
        
        self.conv_layers = torch.nn.Sequential(*layers)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        
        # Fully connected layers
        # fc_input_size = self.num_filters[1] * (7 // (2 ** (self.num_layers - 1))) * (7 // (2 ** (self.num_layers - 1)))  # 입력 크기 계산
        fc_input_size = self.num_filters[1] * 7 * 7  # num_filters[1]의 수에 따라
        self.fc1 = torch.nn.Linear(fc_input_size, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)  # (배치 크기, 채널 수, 높이, 너비)
        x = self.conv_layers(x)
        x = x.view(-1, self.num_flat_features(x))
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.nn.functional.softmax(x, dim=1)
    
    def num_flat_features(self, x):
        size = x.size()[1:] # 배치 차원을 제외한 모든 차원의 크기를 곱함
        num_features = 1
        for s in size:
            num_features *= s
        return num_features