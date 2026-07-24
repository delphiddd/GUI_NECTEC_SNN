import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_OUTPUTS = 26
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' Total MACs per image   : 8,456,666
 Energy/inference (pJ)  : 38,900,663.60 pJ
 Energy/inference (µJ)  : 38.9007 µJ
 Total (train) images   : 2,121,600 uses Energy 82.531648 J
 Total (val)   images   : 374,400   uses Energy 14.564408 J
 Total (test)  images   : 416,000  uses Energy 16.182676 J
 Total images           : 2,912,000  uses Energy 113.278732 J
'''

TOTAL_MACS = 8_456_666
ENERGY_J   = TOTAL_MACS * 4.6e-12  # ~4.6 pJ/MAC (45nm CMOS estimate)

ENG_MAP = {
     0:'a',  1:'b',  2:'c',  3:'d',  4:'e',  5:'f',  6:'g',
     7:'h',  8:'i',  9:'j', 10:'k', 11:'l', 12:'m', 13:'n',
    14:'o', 15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u',
    21:'v', 22:'w', 23:'x', 24:'y', 25:'z'
}


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Block 1: 28x28 → MaxPool → 14x14
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        
        # Block 2: 14x14 → MaxPool → 7x7
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        
        # Block 3: 7x7 → MaxPool → 3x3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(128)

        # Pool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)

        
        # Fully connected
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, NUM_OUTPUTS)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
       
        x = self.dropout(x)

        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def build_model():
    return Net().to(DEVICE)


def load_model(pth_path: str):
    net = build_model()
    net.load_state_dict(torch.load(pth_path, map_location=DEVICE, weights_only=True))
    net.eval()
    return net


def predict(net, img_tensor: torch.Tensor) -> tuple[int, str, list[float], float]:
    """returns (predicted_idx, eng_label, confidence_per_class, energy_joules)"""
    img_tensor = img_tensor.to(DEVICE)

    with torch.no_grad():
        logits = net(img_tensor).squeeze(0)  # [26]

    probs      = F.softmax(logits, dim=0)
    predicted  = probs.argmax().item()
    confidence = probs.tolist()

    return predicted, ENG_MAP[predicted], confidence, ENERGY_J
