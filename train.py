import torch
from torch.autograd.variable import Variable
import numpy as np
from tqdm import tqdm
from torchnet import meter
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, training_generator, n_epochs, total_size, loss,
          optimizer, val_generator, val_size, conv, one_trace, verbose):
    # Loss and optimizer
    i = 0
    validation_losses = []
    training_losses = []
    validation_accuracy = []
    training_accuracy = []
    cms = []
    for epoch in range(n_epochs):
        batch_loss = []
        correct = 0
        total = 0
        for values in tqdm(training_generator):
            '''
            # Move tensors to the configured device
            inp = torch.from_numpy(inp).float().to(device)
            output = torch.from_numpy(output).float().to(device)
            '''
            if(conv and one_trace):
                inp = Variable(values[0]).view(values[0].size(0), 1, values[0].size(1))
            elif(conv):
                inp = Variable(values[0]).view(values[0].size(0), 3, values[0].size(2))
            else:
                inp = Variable(values[0])
            output = Variable(values[1].squeeze()).type(torch.LongTensor)
            
            # Forward pass
            predicted = model(inp)
            
            # softmax to get the probability of the classes
            # predicted = F.softmax(predicted, dim=1)
            _, labels = torch.max(predicted.data, 1)
            total += output.size(0)
            correct += (labels == output).sum().item()

            computed_loss = loss(predicted, output)
            batch_loss.append(computed_loss.item())
            # Backward and optimize
            optimizer.zero_grad()
            computed_loss.backward()
            optimizer.step()
        training_losses.append(np.mean(batch_loss))
        if (i+1) % 30 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, n_epochs, i+1, total_size,
                          computed_loss.item()))

        i += 1
        confusion_matrix = meter.ConfusionMeter(2)  # I have 5 classes here
        val_loss, accuracy, confusion_matrix = test(model, val_generator,
                                                    val_size, loss,
                                                    confusion_matrix=confusion_matrix,
                                                    conv=conv,
                                                    one_trace=one_trace,
                                                    verbose=verbose)
        validation_losses.append(val_loss)
        validation_accuracy.append(accuracy)
        training_accuracy.append((100*correct)/total)
        cms.append(confusion_matrix)
    '''
    confusion_matrix = meter.ConfusionMeter(5)  # I have 5 classes here
    _, _, confusion_matrix = test(model, val_generator, val_size, loss,
                                  confusion_matrix=confusion_matrix,
                                  verbose=verbose)
    cms.append(confusion_matrix)
    '''
    results = {
        "validation_losses": validation_losses,
        "training_losses": training_losses,
        "validation_accuracy": validation_accuracy,
        "training_accuracy": training_accuracy,
        "confusion_matrix": cms
        }
    return model, results


def test(model, test_generator, total_size, loss, confusion_matrix=None,
         conv=True, one_trace=False, verbose=False):
    total_loss = []
    correct = 0
    total = 0
    cm = confusion_matrix
    with torch.no_grad():
        for values in tqdm(test_generator):
            if(conv and one_trace):
                inp = Variable(values[0]).view(values[0].size(0), 1, values[0].size(1))
            elif(conv):
                inp = Variable(values[0]).view(values[0].size(0), 3, values[0].size(2))
            else:
                inp = Variable(values[0])
            output = Variable(values[1].squeeze()).type(torch.LongTensor)
            predicted = model(inp)
            if(cm is not None):
                cm.add(predicted.data.squeeze(),
                       output.type(torch.LongTensor))
            # softmax to get the probability of the classes
            # predicted = F.softmax(predicted, dim=1)
            total_loss.append(loss(predicted, output))
            _, labels = torch.max(predicted.data, 1)
            total += output.size(0)
            correct += (labels == output).sum().item()
    if(verbose):
        print('Mean and standard deviation in dataset with size {} are: {} +- {}'.format(
            total_size, np.mean(total_loss), np.std(total_loss)))
    return np.mean(total_loss), (100*correct)/total, cm
