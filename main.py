import argparse
import time

from read_data import read_traces_CNN, read_traces_NN
from dataset import Dataset
from NN import NeuralNet, ConvNeuralNet2Dense
from train import train

from torch.utils import data
import torch
from torch import nn

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# set seed for weight initialization
torch.manual_seed(1234)

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 12}

# Global Parameters
numItera = 5 # number of iterations to repeat
output_size = 2 # number of classes
units_CNN = [50, 10, 100, 100]
units_NN = [50, 20, 10] 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train convolutional neural network for predicting cosmic air showers")
    parser.add_argument(
        '--conv',
        nargs='*',
        dest='conv',
        default=True,
        help='If we want the conv net',
    )
    parser.add_argument(
        '--name',
        nargs='*',
        dest='name',
        default=True,
        help='Name for plots',
    )
    parser.add_argument(
        '--epochs',
        nargs='*',
        dest='epochs',
        default=True,
        help='Number of epochs',
    )
    parser.add_argument(
        '--onetrace',
        nargs='*',
        dest='one_trace',
        default=True,
        help='If we want to use one or multiple traces',
    )
    parser.add_argument(
        '--onevsall',
        nargs='*',
        dest='one_vs_all',
        default=True,
        help='If we want to do one vs all',
    )

    args = parser.parse_args()
    conv = int(args.conv[0])
    name = str(args.name[0])
    epochs = int(args.epochs[0])
    one_trace = int(args.one_trace[0])
    one_vs_all = int(args.one_vs_all[0])


if(conv):
    # Read traces and labels for CNN
    inputs, labels = read_traces_CNN(path="/home/paquillo/TFM/PredMuones/Datos/Data-paco/", min_distance=500,
                                     max_distance=1000, one_trace=one_trace,
                                     one_vs_all=one_vs_all, balanced=True)
else:
    # Read traces and labels for NN
    inputs, labels = read_traces_NN(path="/home/paquillo/TFM/PredMuones/Datos/Data-paco/", min_distance=500,
                                    max_distance=1000, one_trace=one_trace,
                                    one_vs_all=one_vs_all)

if(one_trace):
    input_channels = 1
else:
    input_channels = 3
for i in range(numItera):
    random_st = 42 * i

    if(conv):
        model = ConvNeuralNet2Dense(input_channels=input_channels,
                                    units=units_CNN,
                                    output_size=output_size)
    else:
        model = NeuralNet(input_size=inputs.shape[1],
                          units=units_NN,
                          output_size=output_size)

    X_train, X_val, y_train, y_val = train_test_split(
                                    inputs, labels, test_size=0.2,
                                    random_state=random_st)
    # w = torch.Tensor([12.0,0.5])
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                                 betas=(0.09, 0.999),  # eps=1e-08,
                                 weight_decay=0.0)   
    dataset_train = Dataset(X_train, y_train)
    training_generator = data.DataLoader(dataset_train, **params)
    dataset_val = Dataset(X_val, y_val)
    val_generator = data.DataLoader(dataset_val, **params)
    print("Training")
    start = time.time()
    model, results = train(model, training_generator, epochs,
                           total_size=X_train.shape[0], loss=loss,
                           optimizer=optimizer,
                           val_generator=val_generator,
                           val_size=X_val.shape[0],
                           conv=conv, one_trace=one_trace,
                           verbose=False)
    end = time.time()
    print("Time elapsed {}".format(end - start))
    print("End Training")

    # getting results values
    cms = results['confusion_matrix']
    validation_accuracy = results["validation_accuracy"]
    validation_losses = results["validation_losses"]
    training_accuracy = results["training_accuracy"]
    training_losses = results["training_losses"]

    max_acc_val = np.where(validation_accuracy == np.max(validation_accuracy))
    cm = cms[max_acc_val[0][0]]
    # printing results
    print("Confusion Matrix: \n")
    print(cm.conf)
    print("\n")

    F1Score = np.zeros(output_size)
    for cls in range(output_size):
        try:
            F1Score[cls] = 2.*cm.conf[cls, cls]/(np.sum(cm.conf[cls, :])+np.sum(cm.conf[:, cls]))
        except:
            pass

    print("F1Score: ")
    for cls, score in enumerate(F1Score):
        print("{}: {:.2f}".format(cls, score))

    print("\n")
    print("Best accuracy in validation {} at epoch {}".format(
        np.max(validation_accuracy), np.argmax(validation_accuracy)+1))

    print("\n")
    print("Best accuracy in training {} at epoch {}".format(
        np.max(training_accuracy), np.argmax(training_accuracy)+1))

    print("\n")
    print("Lower loss in validation {} at epoch {}".format(
        np.min(validation_losses), np.argmin(validation_losses)+1))

    print("\n")
    print("Lower loss in training {} at epoch {}".format(
        np.min(training_losses), np.argmin(training_losses)+1))

    # plotting validation loss
    plt.figure()
    plt.title("Validation loss through epochs")
    plt.xlabel("Nº of epochs")
    plt.ylabel("Validation Loss")
    plt.plot(list(range(0, len(validation_losses))), validation_losses)
    plt.savefig("validation_loss_"+str(i)+"-"+name+".png")

    # plotting training loss
    plt.figure()
    plt.title("Training loss through epochs")
    plt.xlabel("Nº of epochs")
    plt.ylabel("Training Loss")
    plt.plot(list(range(0, len(training_losses))), training_losses)
    plt.savefig("training_loss_"+str(i)+"-"+name+".png")

    # plotting validation accuracy
    plt.figure()
    plt.title("Validation Accuracy through epochs")
    plt.xlabel("Nº of epochs")
    plt.ylabel("Accuracy")
    plt.plot(list(range(0, len(validation_accuracy))), validation_accuracy)
    plt.savefig("validation_accuracy_"+str(i)+"-"+name+".png")
