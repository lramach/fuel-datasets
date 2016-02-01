import gzip
import os
import struct
import h5py
import numpy
from fuel.converters.base import fill_hdf5_file

##Converting
def convert_iris(directory, output_directory, output_filename='iris.hdf5'):
  output_path = os.path.join(output_directory, output_filename)
  h5file = h5py.File(output_path, mode='w')
  classes = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
  data = numpy.loadtxt(
          os.path.join(directory, 'iris.data'),
          converters={4: lambda x: classes[x]},
          delimiter=',')
  numpy.random.shuffle(data)
  features = data[:, :-1].astype('float32')
  targets = data[:, -1].astype('uint8')
  train_features = features[:100]
  train_targets = targets[:100]
  valid_features = features[100:120]
  valid_targets = targets[100:120]
  test_features = features[120:]
  data = (('train', 'features', train_features),
          ('train', 'targets', train_targets),
          ('valid', 'features', valid_features),
          ('valid', 'targets', valid_targets),
          ('test', 'features', test_features))
  fill_hdf5_file(h5file, data)
  h5file['features'].dims[0].label = 'batch'
  h5file['features'].dims[1].label = 'feature'
  h5file['targets'].dims[0].label = 'batch'
  h5file['targets'].dims[1].label = 'index'
  
  h5file.flush()
  h5file.close()
  
  return (output_path,)

def fill_subparser(subparser):
    return convert_iris
