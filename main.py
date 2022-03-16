# module load python3/3.8.11
# module load cudnn/v8.2.0.53-prod-cuda-11.3

import warnings
import os
import torch
import torch.distributions as dists
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
import numpy as np
from netcal.metrics import ECE
from laplace import Laplace
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
warnings.simplefilter("ignore", UserWarning)
writer = SummaryWriter('runs/fashion_mnist_experiment_1')


np.random.seed(7777)
torch.manual_seed(7777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


n_epochs = 1
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
num_classes=10


#TODO Create better data loader with augmentation

train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./files/', train=True, download=True,
                             transform=transforms.Compose([
                               transforms.ToTensor()
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  datasets.FashionMNIST('./files/', train=False, download=True,
                             transform=transforms.Compose([
                               transforms.ToTensor()
                             ])),
  batch_size=batch_size_test, shuffle=True)

targets = torch.cat([y for x, y in test_loader], dim=0).cpu()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 32, 5, 1)
        self.fc1 = nn.Linear(12800, 10)

    #Definer struktur af netværk
    def forward(self,x):
        out = nn.Sequential(
          self.conv1,
          nn.ReLU(), #TODO Maxpooling
          self.conv2,
          nn.ReLU(), #TODO Maxpooling
          self.conv3,
          nn.Flatten(), #TODO GLOBAL Averagepooling
          self.fc1)(x)
        return out

model = CNN()


#Tensorboard
dataiter = iter(train_loader)
images, labels = dataiter.next()
img_grid = utils.make_grid(images)
writer.add_image('four_fashion_mnist_images', img_grid)
writer.add_graph(model, images)
writer.close()



optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()), end="\r")
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      if not os.path.exists("./models"):
        os.mkdir('./models')
      torch.save(model.state_dict(), './models/FashionMNIST_plain.pt')
      if not os.path.exists("./optimizer"):
        os.mkdir('./optimizer')
      torch.save(optimizer.state_dict(), './optimizer/optimizer.pth')



def test():
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = model(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()

@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x))
        else:
            py.append(torch.softmax(model(x), dim=-1))

    return torch.cat(py).cpu()

probs_map = predict(test_loader, model, laplace=False)
acc_map = (probs_map.argmax(-1) == targets).float().mean()
ece_map = ECE(bins=10).measure(probs_map.numpy(), targets.numpy())
nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()
print(f'[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}')

# Laplace
print(1)
la = Laplace(model, 'classification',
             subset_of_weights='last_layer',
             hessian_structure='diag')
print(2)
la.fit(train_loader)
print(5)
la.optimize_prior_precision(method='marglik')
print(1)
probs_laplace = predict(test_loader, la, laplace=True)
print(3)
acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
print(1)
ece_laplace = ECE(bins=10).measure(probs_laplace.numpy(), targets.numpy())
print(1)
nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()
print(f'[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}')

print("All done")


# Use kwags for calibration method specific parameters
def test2(calibration_method=None, **kwargs):
  preds = []
  labels_oneh = []
  correct = 0
  model.eval()
  print("We did it")
  with torch.no_grad():
      for data in test_loader:
          images, labels = data[0], data[1]
          print("1")
          pred = model(images)
          
          if calibration_method:
            pred = calibration_method(pred, kwargs)

          # Get softmax values for net input and resulting class predictions
          sm = nn.Softmax(dim=1)
          pred = sm(pred)

          _, predicted_cl = torch.max(pred.data, 1)
          pred = pred.cpu().detach().numpy()

          # Convert labels to one hot encoding
          label_oneh = torch.nn.functional.one_hot(labels, num_classes=num_classes)
          label_oneh = label_oneh.cpu().detach().numpy()

          preds.extend(pred)
          labels_oneh.extend(label_oneh)

          # Count correctly classified samples for accuracy
          correct += sum(predicted_cl == labels).item()
  print("We did it")
  preds = np.array(preds).flatten()
  labels_oneh = np.array(labels_oneh).flatten()

  correct_perc = correct / len(test_loader)
  print('Accuracy of the network on the test images: %d %%' % (100 * correct_perc))
  print(correct_perc)
  
  return preds, labels_oneh

preds, labels_oneh = test2()
print(labels_oneh)
print(preds)

def calc_bins(preds):
  # Assign each prediction to a bin
  num_bins = 10
  bins = np.linspace(0.1, 1, num_bins)
  binned = np.digitize(preds, bins)

  # Save the accuracy, confidence and size of each bin
  bin_accs = np.zeros(num_bins)
  bin_confs = np.zeros(num_bins)
  bin_sizes = np.zeros(num_bins)

  for bin in range(num_bins):
    bin_sizes[bin] = len(preds[binned == bin])
    if bin_sizes[bin] > 0:
      bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
      bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

  return bins, binned, bin_accs, bin_confs, bin_sizes




from visualization.plot import * #TODO FIX PLOTS


def T_scaling(logits, args):
  temperature = args.get('temperature', None)
  return torch.div(logits, temperature)


temperature = nn.Parameter(torch.ones(1).cpu())
args = {'temperature': temperature}
criterion = nn.CrossEntropyLoss()

# Removing strong_wolfe line search results in jump after 50 epochs
optimizer = optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')

logits_list = []
labels_list = []
temps = []
losses = []

for i, data in enumerate(test_loader):
    images, labels = data[0], data[1]

    model.eval()
    with torch.no_grad():
      logits_list.append(model(images))
      labels_list.append(labels)

# Create tensors
logits_list = torch.cat(logits_list).cpu()
labels_list = torch.cat(labels_list).cpu()

def _eval():
  loss = criterion(T_scaling(logits_list, args), labels_list)
  loss.backward()
  temps.append(temperature.item())
  losses.append(loss)
  return loss


optimizer.step(_eval)

print('Final T_scaling factor: {:.2f}'.format(temperature.item()))
"""
plt.subplot(121)
plt.plot(list(range(len(temps))), temps)

plt.subplot(122)
plt.plot(list(range(len(losses))), losses)
plt.show()"""



preds_original, _ = test2()
preds_calibrated, _ = test2(T_scaling, temperature=temperature)

draw_reliability_graph(preds_original,"soft_")
draw_reliability_graph(preds_calibrated,"lap_")