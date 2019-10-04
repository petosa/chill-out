from search.bfs import BFS
from policyenv import PolicyEnv
from squeezenet import squeezenet1_1
from alexnet import alexnet
import util
'''
model = squeezenet1_1(pretrained=True)
count = 0
for i, (layer, w) in enumerate(model.named_children()):
    for j, (layer1, w1) in enumerate(w.named_children()):
        print(j)
        for k, (layer2, w2) in enumerate(w1.named_parameters()):
            print(layer, layer1, layer2)
            if j == 4:
                w2.requires_grad = False
            print(w2.requires_grad)
            count += 1

print(util.get_trainable_layer_count(model))
print(count/2)

quit()
'''
env = PolicyEnv(num_layers=8)
search = BFS(env)

s, v = None, None
iters = 0
for _ in range(1000000):
    if iters % 5000 == 0:
        print(iters, len(search.frontier))
    iters += 1
    pair = search.next()
    if pair is None: break
    state, val = pair
    if s is None or val < v:
        s, v = state, val

print(iters, "iter", s, "state", v, "val")
for t in search.trace(s):
    print(t, "\n")
