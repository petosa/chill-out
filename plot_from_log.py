import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

regex = r'[0-9]+\,[0-9]*\.?[0-9]+\,[0-9]+\,[0-9]*\.?[0-9]+\,[0-9]+'
loss_line = re.compile(regex)
files = ['unfrozen.log', 'chain_thaw.log', 'gradual_unfreezing.log']
#files = [ 'small_gradual_unfreezing.log', "small_unfrozen.log"]

fig = plt.figure()

for i, filename in enumerate(files):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    with open('logs/' + filename, 'r') as f:
        for line in f:
            if loss_line.match(line):
                line = line[:-1]
                _, train_loss, train_acc, valid_loss, valid_acc = [float(x) for x in line.strip().split(',')]
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                val_losses.append(valid_loss)
                val_accs.append(valid_acc)
    plt.plot(val_losses, label=filename.split('.')[0])

plt.title('All Loss')
plt.ylabel('loss')
#plt.ylim((0, 1))
plt.legend(loc = 'upper left')
fig.savefig('all_loss.png')

# plt.plot(train_losses, label='train', color='blue')
# plt.plot(val_losses, label='val', color='orange')
# plt.title('Chain Thaw All Loss')
# plt.ylabel('loss')
# plt.ylim((0, 1))
# plt.legend(loc = 'upper left')
# fig.savefig('all_loss.png')

# fig = plt.figure()
# plt.plot(train_accs, label='train', color='blue')
# plt.plot(val_accs, label='val', color='orange')
# plt.title('Chain Thaw All Accuracy')
# plt.ylabel('accuracy')
# plt.legend(loc = 'lower left')
# fig.savefig('all_acc.png')
