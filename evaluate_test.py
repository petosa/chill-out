import train
import util
import torch


print("Custom")
t= train.Trainer(0, None)
m = util.load_alexnet(10).cuda()
o = torch.optim.SGD(m.parameters(),1e-4,.9,weight_decay=1e-4,nesterov=True)
util.full_load(m,o,133,1572362438)

print(t.evaluate(m, "train"))
print(t.evaluate(m, "val"))
print(t.evaluate(m, "test"))


print("Unfreezing")
m = util.load_alexnet(10).cuda()
o = torch.optim.SGD(m.parameters(),1e-4,.9,weight_decay=1e-4,nesterov=True)
util.full_load(m,o,9,1572359480)

print(t.evaluate(m, "train"))
print(t.evaluate(m, "val"))
print(t.evaluate(m, "test"))


print("Gradual unfreeze")
m = util.load_alexnet(10).cuda()
o = torch.optim.SGD(m.parameters(),1e-4,.9,weight_decay=1e-4,nesterov=True)
util.full_load(m,o,8,1572359851)

print(t.evaluate(m, "train"))
print(t.evaluate(m, "val"))
print(t.evaluate(m, "test"))


print("Chain thaw")
m = util.load_alexnet(10).cuda()
o = torch.optim.SGD(m.parameters(),1e-4,.9,weight_decay=1e-4,nesterov=True)
util.full_load(m,o,9,1572360442)

print(t.evaluate(m, "train"))
print(t.evaluate(m, "val"))
print(t.evaluate(m, "test"))
