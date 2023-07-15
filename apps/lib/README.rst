DLCJob Library
=================
All datasets that represent a map from keys to data samples should subclass
the DLCJobDataset class. All subclasses should overwrite: 

* ``__process__```: supporting pre-processing loaded data. Data are saved as key-value map before calling this method. You are responsible for reshaping the dict to desired array.
* ``__getitem__``: supporting fetching a data sample for a given index.
* ``__len__``: returning the size of the dataset

Installing
============

.. code-block:: bash

    pip3 install dlcjob

Usage
=====

.. code-block:: python

    from DLCJob import DLCJobDataset, DLCJobDataLoader
    
    # ImageNetDataset Class Example:
    class ImageNetDataset(DLCJobDataset):
        def __init__(keys, ...):
            super().__init__(keys)
            ...
        
        def __process__(self):
            ...
        
        def __getitem__(self):
            ...
        
        def __len__(self):
            ...
    
    # In your main program
    val_dataset = ImageNetDataset(keys=['imagenet-mini/val'], transform=transform)
    ...
    val_loader = DLCJobDataLoader(val_dataset, ...)
    
