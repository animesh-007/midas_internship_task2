import torch
import torch.nn as nn
import torch.nn.functional as F


# 28 ke shape ke liye tha    
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1,1) 
        self.conv2 = nn.Conv2d(16, 32, 3, 1,1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        #self.fc1 = nn.Linear(9216, 128)
        self.fc1 = nn.Conv2d(32,128,1,1) #nn.Linear(9216, 128)
        self.fc2 = nn.Conv2d(128,62,1,1) #nn.Linear(128, 62)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
#        x = torch.flatten(x, 1)
        x = F.adaptive_avg_pool2d(x,(1,1))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        output = output.view(-1,62)
        return output

# 28 ke shape ke liye tha  yeh bhi
class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1,1) 
        self.conv2 = nn.Conv2d(16, 32, 3, 1,1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        #self.fc1 = nn.Linear(9216, 128)
        self.fc1 = nn.Linear(32,128) #nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128,62) #nn.Linear(128, 62)
    
    
    def forward(self, x):
#        import pdb; pdb.set_trace()
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
#        
        x = F.adaptive_avg_pool2d(x,(1,1))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
#        output = output.view(-1,62)
        return output

class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1,1) 
        self.conv2 = nn.Conv2d(16, 32, 3, 1,1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1,1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1,1)  # [1, 128, 22, 22]
        self.conv5 = nn.Conv2d(128, 128, 3, 1,1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        #self.fc1 = nn.Linear(9216, 128)
#        self.fc1 = nn.Conv2d(32,128,5,1) #nn.Linear(9216, 128)
        self.fc2 = nn.Conv2d(128,62,1,1) #nn.Linear(128, 62)

    def forward(self, x):
#        import pdb; pdb.set_trace()
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.adaptive_avg_pool2d(x,(1,1))
        x = self.fc2(x)
        x = torch.flatten(x, 1)
        output = F.log_softmax(x, dim=1)
        return output
