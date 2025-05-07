import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def start_server(host='xxx.xx.xx.xxx', port=5000):                      # IP hidden
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}")

    model = CNN()

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address}")

        # Receive model weights
        data = b""
        while True:
            packet = client_socket.recv(4096)
            if not packet: break
            data += packet

        client_weights = pickle.loads(data)
        print("Received model weights")

        model.load_state_dict(client_weights)
        torch.save(model.state_dict(), "global_model.pth")

        client_socket.send("Model received".encode('utf-8'))
        client_socket.close()

if __name__ == '__main__':
    start_server()
