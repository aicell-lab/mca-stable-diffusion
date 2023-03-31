from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from ldm.data.hpa import HPACombineDatasetMetadataInMemory
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score


test_dataset = HPACombineDatasetMetadataInMemory(group='validation', seed=123, train_split_ratio=0.95, include_location=True, filter_func="has_location", return_info=True, cache_file="/data//wei/hpa-webdataset-all-composite/HPACombineDatasetMetadataInMemory-256.pickle")
train_dataset = HPACombineDatasetMetadataInMemory(group='train', seed=123, train_split_ratio=0.95, include_location=True, filter_func="has_location", return_info=True, cache_file="/data//wei/hpa-webdataset-all-composite/HPACombineDatasetMetadataInMemory-256.pickle")


def data_generator(dataset, batch_size=16):
    total_batch = len(dataset) // batch_size
    for b in range(total_batch):
        X = []
        y = []
        for i in range(batch_size):
            X.append(dataset[i]['bert'])
            y.append(dataset[i]['location_classes'])
        yield torch.from_numpy(np.stack(X, axis=0)), torch.from_numpy(np.stack(y, axis=0))
    
# g = data_generator(train_dataset)
# X, y = next(g)
# print(X.shape, y.shape)

def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro', zero_division=0),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro', zero_division=0),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples', zero_division=0),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples', zero_division=0),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples', zero_division=0),
            }


class FCNet(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=2048),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=2048, out_features=1024),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1024, out_features=n_classes),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        return self.fc(x)
 

def checkpoint_save(model, optimizer, save_path, epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
    

def train(model, device="cpu", test_freq=10, epochs=100, batch_size=16, save_freq=10, max_epoch_number=1000, save_path="model.pt"):
    epoch = 0
    iteration = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    while True:
        batch_losses = []
        gtrain = data_generator(train_dataset, batch_size=batch_size)
        total_batch = len(train_dataset) // batch_size
        for b in range(total_batch):
            X, y = next(gtrain)
            X, y = X.to(device), y.to(device)
    
            optimizer.zero_grad()
    
            model_result = model(X)
            loss = criterion(model_result, y.type(torch.float))
    
            batch_loss_value = loss.item()
            loss.backward()
            optimizer.step()
    
            batch_losses.append(batch_loss_value)
    
            if iteration % test_freq == 0:
                model.eval()
                with torch.no_grad():
                    model_result = []
                    targets = []
                    test_batches = len(test_dataset) // batch_size
                    gtest = data_generator(test_dataset, batch_size=batch_size)
                    for i in range(test_batches):
                        embed, gt = next(gtest)
                        embed = embed.to(device)
                        model_batch_result = model(embed)
                        model_result.extend(model_batch_result.cpu().numpy())
                        targets.extend(gt.cpu().numpy())
    
                result = calculate_metrics(np.array(model_result), np.array(targets), threshold=0.3)
                print("epoch:{:2d} iter:{:3d} test: "
                    "micro f1: {:.3f} "
                    "macro f1: {:.3f} "
                    "samples f1: {:.3f}".format(epoch, iteration,
                                                result['micro/f1'],
                                                result['macro/f1'],
                                                result['samples/f1']))
    
                model.train()
            iteration += 1
    
        loss_value = np.mean(batch_losses)
        print("epoch:{:2d} iter:{:3d} train: loss:{:.3f}".format(epoch, iteration, loss_value))
        if epoch % save_freq == 0:
            checkpoint_save(model, optimizer, save_path, epoch)
        epoch += 1
        if max_epoch_number < epoch:
            break

# Initialize the model
model = FCNet(768, 34)
# yp = model(torch.from_numpy(X))
# print(yp)
train(model)