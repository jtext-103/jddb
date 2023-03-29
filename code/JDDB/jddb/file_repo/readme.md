# How to use FileRepo 

**`file_repo`** is a subpackage used to process HDF5 files.

It has a class named **`FileRepo`**.

## Definition of the file format

Shot file should *always* named as **'shot_no.hdf5'** (such as 1050000.hdf5).

Each file contain two groups, **'data'** and **'meta'**.

- 'Data' group:

   - Each *tag* (diagnostic raw data or processed data) should save in the **'data'** group as a dataset.

   - **Attributes** belong to each dataset, and by default, they should include the sampling rate named as **'SampleRate'** and start time named as **'StartTime'** of the data for reconstructing the time axis.
 
- 'Meta' group:

  - Each *label* should save in the **'meta'** group as a dataset.
  - No **attribute** belongs to the dataset in **'meta'** group. 
  - The labels can be synced from MongoDB service to the hdf5 file using MetaDB. 
  
## How file is organized

All the files should be under a base folder.

The base folder should use $shot_x$ for template.

## FileRepo

- **`FileRepo(base_path)`**

  base path contains template same as BM design `\\data\\jtext\\$shot_2$XX\\$shot_1$XX\\`
  
  Example:
  
  ```python
  from jddb.file_repo import FileRepo
  base_path = "\\data\\jtext\\$shot_2$XX\\$shot_1$XX\\"
  file_repo = FileRepo(base_path)
  ```

### Operate files

- **`get_file(shot_no, ignore_none=False)`**

  Get file path for one shot, if not exist return empty string.
  - `ignore_none` controls the return value, if `ignore_none=True`, even the shot file does exist, still return the file path, not the empty string.
    Example:

  ```python
  shot_no = 1050000
  file_path = file_repo.get_file(shot_no)
  ```

- **`get_all_shots()`**
  
  Find all shots in the base path, return a shot list.
  
  Example:
  ```python
  all_shot_list = file_repo.get_all_shot()
  ```

- **`create_shot(shot_no)`**

  Create the a shot file, return the file path.
  
  Example:
  
  ```python
  shot_no = 1050001
  file_path = file_repo.create_shot(shot_no)
  ```
  

- **`get_files(shot_list=None,create_empty=False)`**

  Get files path for a shot list, return a dictionary `dict{'shot_no': file_path}` 
  - `shot_list=None` If no value is assigned to `shot_list`, return all the shot list in the root path. `shot_list = file_repo.get_all_shot()`.
  - `create_empty` controls if you want to create new file for the shot with no file exist. If `create_empty=True`, the return dictionary will contains no empty string, and an empty shot file will be created.

  Example:
  ```python
  shot_list = [1050000, 1050001]
  files_path = file_repo.get_files(shot_list)
  ```
### Read
- **`get_tag_list(shot_no)`**

  Get all the tag list of the data group in one shot file, return a tag list.
  
  Example:
  ```python
  shot_no = 1050002
  tag_list = file_repo.get_tag_list(shot_no)
  ```
- **`read_data_file(file_path, tag_list=None)`**

  Read data dict from the data group in one shot file with a file path as input, return a dictionary `dict{'tag': data}`.
  - `tag_list=None` If no list is assigned to `tag_list`, return the dictionary with all the tag list in this shot file. `tag_list = file_repo.get_tag_list(shot_no)`.
  
  Example:

  ```python
  file_path = "\\data\\jtext\\10500XX\\105000X\\1050000.hdf5"
  tag_list = ['\\ip', '\\bt']
  data_dict = file_repo.read_data_file(file_path, tag_list)
  ```
- **`read_data(shot_no, tag_list=None)`**

  Read data dict from the data group in one shot file with a shot number as input, return a dictionary `dict{'tag': data}`.
  - `tag_list=None` If no list is assigned to `tag_list`, return the dictionary with all the tags in this data group. 
  
  Example:

  ```python
  shot_no = 1050000
  tag_list = ['\\ip', '\\bt']
  data_dict = file_repo.read_data(shot_no, tag_list)
  ```
- **`read_attributes(shot_no, tag, attribute_list=None)`**

  Read attribute dict of one tag in one shot file, return a dictionary `dict{'SampleRate': 1000}`.
  
  - `attribute_list=None` If no list is assigned to `attribute_list`, return the dictionary with all the attributes in this dataset.
  
  Example:
  
  ```python
  shot_no = 1050000
  tag = '\\ip'
  attribute_list = ['SampleRate', 'StartTime']
  attribute_list = file_repo.read_attributes(shot_no, tag, attribute_list)
  ```
  
- **`read_labels_file(file_path, label_list=None)`**
  
  Read label dict from the meta group in one shot file with a file path as input, return a dictionary `dict{'DownTime': 0.6}`.
  
  - `label_list=None` If no list is assigned to `label_list`, return the dictionary with all the labels in this meta group.
  
  Example:
  ```python
  file_path = "\\data\\jtext\\10500XX\\105000X\\1050000.hdf5"
  label_list = ['DownTime', 'IsDisrupt']
  label_dict = file_repo.read_labels_file(file_path, label_list)
  ```

- **`read_labels(shot_no, label_list=None)`**

  Read label dict from the meta group in one shot file with a shot number as input, return a dictionary `dict{'DownTime': 0.6}`.
  
  - `label_list=None` If no list is assigned to `label_list`, return the dictionary with all the labels in this meta group.
  
  Example:
  ```python
  shot_no = 1050000
  label_list = ['DownTime', 'IsDisrupt']
  label_dict = file_repo.read_labels(shot_no, label_list)
  ```
### Remove

- **`remove_data_file(file_path, tag_list)`**

  Remove the datasets from the data group in one shot file with fa ile path as input, return None.
  
  Example:
  ```python
  file_path = "\\data\\jtext\\10500XX\\105000X\\1050000.hdf5"
  tag_list = ['\\ip', '\\bt']
  file_repo.remove_data_file(file_path, tag_list)
  ```
  
- **`remove_data(shot_no, tag_list)`**

  Remove the datasets from the data group in one shot file with a shot number as input, return None.
  
  Example:
  ```python
  shot_no = 1050000
  tag_list = ['\\ip', '\\bt']
  file_repo.remove_data(shot_no, tag_list)
  ```
  
- **`remove_attribute(shot_no, tag, attribute_list)`**

  Remove the attribute of one tag in one shot file, return None.
  
  Example:
  ```python
  shot_no = 1050000
  tag = '\\ip'
  attribute_list = ['StartTime', 'SampleRate']
  file_repo.remove_attributes(shot_no, tag, attribute_list)
  ```  
  
- **`remove_labels_file(file_path, label_list)`**
  
  Remove labels from the meta group in one shot file with a file path as input, return None.
  
  Example:
  ```python
  file_path = "\\data\\jtext\\10500XX\\105000X\\1050000.hdf5"
  label_list = ['DownTime', 'IsDisrupt']
  file_repo.remove_labels_file(file_path, label_list)
  ```
  
- **`remove_labels(shot_no, label_list)`**

  Remove labels from the meta group in one shot file with a shot number as input, return None.
  
  Example:
  ```python
  shot_no = 1050000
  label_list = ['DownTime', 'IsDisrupt']
  file_repo.remove_labels(shot_no, label_list)
  ```
### Write
- **`write_data_file(file_path, data_dict, overwrite=False)`**

  Write a data dictionary in the data group in one shot file with a file path as input, return None.
  
  - `overwrite` controls if the input data will overwrite the original data.
  
  Example:
  ```python
  file_path = "\\data\\jtext\\10500XX\\105000X\\1050000.hdf5"
  data_dict = {'\\ip': np.random.randn(1000), '\\bt': np.random.randn(1000)}
  file_repo.write_data_file(file_path, data_dict)
  ```
- **`write_data(shot_no, data_dict, overwrite=False, create_empty=True)`**

  Write a data dictionary in the data group in one shot file with a shot number as input, return None.
  
  - `overwrite` controls if the input data will overwrite the original data.
  
  - `create_empty` controls if you would like to create a new shot file.
  
  Example:
  ```python
  shot_no = 1050000
  data_dict = {'\\ip': np.random.randn(1000), '\\bt': np.random.randn(1000)}
  file_repo.write_data(shot_no, data_dict)
  ```  

- **`write_attribute(shot_no, tag, attribute_dict, overwrite=False)`**

  Write a attribute dictionary in one tag in one shot file, return None.
  
  - `overwrite` controls if the input data will overwrite the original data.
  
  Example:
  ```python
  shot_no = 1050000
  tag = '\\ip'
  attribute_dict = {'SampleRate': 1000, 'StartTime': 0.1}
  file_repo.write_attribute(shot_no, tag, attribute_dict)
  ```
  
- **`write_label_files(file_path, label_dict, overwrite=False)`**
  
  Write a label dictionary in the meta group in one shot file with a file path as input, return None.
  
  - `overwrite` controls if the input data will overwrite the original data.
  
  Example:
  ```python
  file_path = "\\data\\jtext\\10500XX\\105000X\\1050000.hdf5"
  label_dict = {'DownTime': 0.6, 'IsDisrupt': 1} 
  file_repo.write_label_file(file_path, label_dict)
  ```
  
- **`write_label(shot_no, label_dict, overwrite=False)`**
  
  Write a label dictionary in the meta group in one shot file with a shot number as input, return None.
  
  - `overwrite` controls if the input data will overwrite the original data.
  
  Example:
  ```python
  shot_no = 1050000
  label_dict = {'DownTime': 0.6, 'IsDisrupt': 1} 
  file_repo.write_label_file(shot_no, label_dict)
  ```
### Sync Meta

- **`sync_meta(meta_db: MetaDB, shot_list=None, overwrite=False)`**

  Sync labels to the meta group of the shot file from MetaDB.
  
  - `MetaDB` a class from the module `meta_db.py`
  - `shot_list=None` If no value is assigned to `shot_list`, return all the shot list in the root path. `shot_list = file_repo.get_all_shot()`.
  - `overwrite` controls if the input data will overwrite the original data.
  
  Example:
  ```python
  from jddb.meta_db import MetaDB
  shot_list = [1050000, 1050001]
  connection_str = {
    "host": "1.1.1.1",
    "port": 24011,
    "username": "User",
    "password": "******",
    "database": "JDDB"
  }
  collection = "label"
  meta_db = MetaDB(connection_str, collection)
  file_repo.sync_meta(meta_db, shot_list)
  meta_db.disconnect()
  ```  
