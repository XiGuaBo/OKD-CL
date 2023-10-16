import json
import os,sys
sys.path.append('..')
import utils.dataloader as dl

# f = open("sub_cat.json",'r')
# sc = json.loads(f.read())
# f.close()

sub_set = dl.sub_set

f = open("idx2name.json",'r')
scd = json.loads(f.read())
f.close()

cc = {}
for cidx in sub_set:
    mc = scd[cidx].split(',')[0]
    print (mc)
    if (mc in cc.keys()):
        cc[mc] += 1
    else:
        cc[mc] = 1

print (cc)