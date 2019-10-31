import train
import util
import torch
import sys
import os
import shutil

session, ckpt = sys.argv[1].split("/")
session = int(session)
ckpt = int(ckpt.split(".")[0])
print("Custom")
t = train.Trainer(0, None)
m = util.load_alexnet(10).cuda()
o = torch.optim.SGD(m.parameters(),1e-4,.9,weight_decay=1e-4,nesterov=True)
util.full_load(m,o,ckpt,session)

print(t.evaluate(m, "train"))
print(t.evaluate(m, "val"))
print(t.evaluate(m, "test"))
shutil.rmtree("0")
