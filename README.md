# fuel-datasets
Code for creating, downloading and converting new fuel datasets

Using existing fuel code to generate new datasets by creating train-valid-test splits from existing hdf5 files.

Using Fuel's v0.0.1 from git://github.com/mila-udem/fuel.git@v0.0.1 to update, since that is the version used by ladder networks, which is what the new dataset is being used for.

The code for converting and generating a "NewDataset" fuel type is in converters/new_dataset.py. 
The convert_data method can be modified to fit the nature of thew new dataset being created.

