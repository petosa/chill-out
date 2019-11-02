import util
import sys

session, ckpt = sys.argv[1].split("/")
ckpt = int(ckpt.split(".")[0])

t = util.make_trainer(session)
m = util.make_model()
util.full_load(m,ckpt,session)

print("Train:", t.evaluate(m, "train"))
print("Val:", t.evaluate(m, "val"))
print("Test:", t.evaluate(m, "test"))
