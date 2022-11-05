import os
file = open('/data_a11/mayi/project/ECAPATDNN-analysis/sub_vox2.txt','r')
scp = open('/data_a11/mayi/project/wespeaker/examples/voxceleb/v2/data/sub_vox2_clean/wav.scp', 'w')
keys = []
line = file.readline()
while line:
    subs = line.split(' ')
    if subs[1] not in keys:
        keys.append(subs[1])
    if subs[2] not in keys:
        keys.append(subs[2].strip())
    line = file.readline()

for i in keys:
    spk = i.split('/')[0]
    idx = int(spk[2:])
    num = str(idx//200)
    target = os.path.join('/data_a11/mayi/dataset/VoxCeleb_latest/VoxCeleb2/dev/aac_split/',num,i)
    print(target)
    scp.write(i+' '+target+'\n')
