# How to use MDSDupmer 

**`mds_dumper`** is a subpackage used to dump data from MDSplus (only support MDSpuls now).

It has a class named **`MDSDumper`**.

YOU MUST install **`MDSplus`** package FIRST

## MDSDumper

- **`MDSDumper(host_name, tree_name)`**
  
  Example:
  ```python
  from jddb.file_repo import MDSDumper
  dump = MDSDumper('1.1.1.1', 'jtext')
  ```
### Functions
- **`connect`**
  
  Connect to the host

- **`disconnect`**

  Disconnect to the host

- **`dumper(file_repo: FileRepo, shot_list, tag_list, overwrite=False)`**

  Dump data from MDSPlus into the hdf5 shot file.
  
  - `FileRepo` a class from the module `file_repo.py`
  - `overwrite` controls if the input data will overwrite the original data.
  
  Example:
  ```python
  from jddb.file_repo import FileRepo
  from jddb.mds_dumper import MDSDumper
  base_path = "\\data\\jtext\\$shot_2$XX\\$shot_1$XX\\"
  file_repo = FileRepo(base_path)
  shot_list = [1050000, 1050001]
  tag_list = ['\\ip', '\\bt']
  dump = MDSDumper('1.1.1.1', 'jtext')
  dump.dumper(file_repo, shot_list, tag_list) 
  ```
