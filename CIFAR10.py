import torch
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
import numpy as np
import MSD_CSC_ISTA as msdcsc
import torch.nn.functional as F
from torch.autograd import Variable

EPOCH = 300
BATCH_SIZE = 64
cudaopt = True
# prepare data
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transform_test)

train_loader = Data.DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle = True) #sampler=Data.sampler.SubsetRandomSampler(train_indices))
test_loader = Data.DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=True)

# record test accuracy
Acc_test = np.zeros((EPOCH,))

torch.cuda.set_device(1)
# define model
#model = msdcscista.MSD_CSC_ISTA(64,4,128,128,128*7*7,3,10,5,1)
model = msdcsc.MSD_CSC_net(12,12,3,10)
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# train
current_Acc = 0
for epoch in range(EPOCH):
    model.train()
    for step, (x, y) in enumerate(train_loader):
        print(step)
        b_x = Variable(x)
        b_y = Variable(y)
        if cudaopt:
            b_y, b_x = b_y.cuda(), b_x.cuda()
        scores = model(b_x)
        loss = F.nll_loss(scores, b_y)  # negative log likelyhood
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        model.zero_grad()

    # testing
    model.eval()
    correct = 0
    test_loss = 0
    for step, (x, y) in enumerate(test_loader):
        b_x = Variable(x)
        b_y = Variable(y)  # batch label
        if cudaopt:
            b_y, b_x = b_y.cuda(), b_x.cuda()
        scores = model(b_x)
        test_loss += F.nll_loss(scores, b_y, reduction='sum').data.item()
        pred = scores.data.max(1, keepdim=True)[1]
        correct += pred.eq(b_y.data.view_as(pred)).long().cpu().sum()
        # print(step)

    Acc_test[epoch] = 100 * float(correct) / float(len(test_loader.dataset))

    print('Epoch: ', epoch, '| test acc: ', Acc_test[epoch], '%')
    # save parameters
    if current_Acc < Acc_test[epoch]:
        current_Acc = Acc_test[epoch]
        torch.save(model.state_dict(), "./results/CIFAR10MSDCSC_model"+str(current_Acc)+".pth")
# save the Acc_test records
np.save("./results/live/CIFAR10MSDCSC_Acc_test.npy", Acc_test)

print("train done!")