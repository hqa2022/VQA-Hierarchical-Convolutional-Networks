import h5py

path = '../../data/activity-net/sub_activitynet_v1-3.c3d.hdf5'

dataset = h5py.File(path, 'r')
keys = list(dataset.keys())

all_num = 0
for i in range(len(keys)):
    feature = dataset[keys[i]]['c3d_features']
    all_num += feature.shape[0]
    f = h5py.File('../../data/activity-c3d/'+keys[i]+'.h5', 'w')
    f.create_dataset('feature', data=feature)
    f.close()
    if i % 500 == 0:
        print(i)


print('avg frame number: ', all_num/i)

