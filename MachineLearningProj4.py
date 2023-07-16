import backprop_data

import backprop_network

import matplotlib.pyplot as plt

training_data, test_data = backprop_data.load(train_size=10000, test_size=5000)

net = backprop_network.Network([784, 40, 10])

net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

learning_rates = [0.001, 0.01, 0.1, 1, 10, 100]
epochs = 30
mini_batch_size = 10
train_acc = []
train_loss = []
test_acc = []

# Perform training with different learning rates
for learning_rate in learning_rates:
    net = backprop_network.Network([784, 40, 10])
    cur_train_acc, cur_train_loss, cur_test_acc = net.SGD_b(training_data, epochs=epochs,mini_batch_size=mini_batch_size,learning_rate=learning_rate,test_data=test_data)
    train_acc.append(cur_train_acc)
    train_loss.append(cur_train_loss)
    test_acc.append(cur_test_acc)
    print("finished learning rate {}".format(learning_rate))

#Q1b
for index, rate in enumerate(learning_rates):
    plt.plot(range(epochs), train_loss[index], label="rate = {}".format(rate))

plt.title("Training Loss for The Learning Rates: 0.001, 0.01, 0.1, 1, 10, 100")
plt.xlabel('Epochs')
plt.ylabel("Loss")
plt.legend()
plt.show()

for index, rate in enumerate(learning_rates):
    plt.plot(range(epochs), train_acc[index], label="rate = {}".format(rate))

plt.title("Training Accuracy for The Learning Rates: 0.001, 0.01, 0.1, 1, 10, 100")
plt.xlabel('Epochs')
plt.ylabel("Accuracy")
plt.legend()
plt.show()

for index, rate in enumerate(learning_rates):
    plt.plot(range(epochs), test_acc[index], label="rate = {}".format(rate))

plt.title("Test Accuracy for The Learning Rates: 0.001, 0.01, 0.1, 1, 10, 100")
plt.xlabel('Epochs')
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Q1c
training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
net = backprop_network.Network([784, 40, 10])
net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

# Q1d
training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
net = backprop_network.Network([784, 250, 10])
net.SGD(training_data, epochs=55, mini_batch_size=11, learning_rate=0.25, test_data=test_data)
