# How to use FileRepo (沈总)

这里就是读写文件的，这个东西依赖 MetaDB，当然只有一个功能依赖，就是 download meta

shotfile always named as shotNo.hdf5

## the file format

how file is formatted, like group, attribute and etc.

## How file is organized

shotfile always named as shot.h5

all under a bast folder

use $shot_x$ for template

上面的写完整

## Conntect to file repo

- 构造函数(base_path)
  base path contains template same as BM design "\data\jtext\$shot_2$00\"

## Read

- get_file(shot_no)
  return the file path for that shot
  if not exist return empty string
  Example:

  ```python
  # add examples here
  ```
- get_files(shot_list=none,create_empty=false)


- read_data_file(file_path,tags)
- read_data(shot,tags)
- read_labels_file(file_path)
- read_labels(shot)
- next_shot(reset=false,start_shot=none)
  iterate all the shot in the repo, return next shot file path
- count()
  shot count in this repo

## Write

- create_shot(shot_no)
  create a empty shot file

- put_label_file(shot_file,labels)
  给某一跑写入一些 label，label 是一个字典，key 是 label， value 是 label 的值
- put_label(shot_no,labels)

- wirte_date_file(file_path,data,attributes)  
  data is same format read.
- wirte_date(shot,data,attributes)

## Sync Meta

sync_meta(meta_db,shot_list=None, overwrite=fasle)
sync label from MetaDB,
overwrite=false not overwrite existing label in file repo
shot_list=none means sync every shot in the file repo

## Dump

now only support MDSplus

here make another class: MDSDumper

MDSDumper.Connect(host,tree)

MDSDumper.Dump(file_repo,shot_list,tag_list)

