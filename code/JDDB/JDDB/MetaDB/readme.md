# How to use MetaDB

这里说一下明明，原来 tag 不清晰，现在明确一下 tag 值得是诊断的 tag，在 MetaDB 里面哪些 key value pair 叫做 label

读写文件的都放在 FileRepo 里面了

## Conntect to db

- connect
  connect(connection_string,collection)
  connection string is the connections string to a mongodb server. collection is the collection for the MetaDB.  
  connect befor any other action

- disconnect
  as name suggests.

## Query

- get_labels(shot_no)
  just like db.tag(shot)

  Example:

  ```python
  # add examples here
  ```

- query(shot_list=None,filter=None)
  just like db.query(shotlist=None, filter=None)

- query_valid(shotlist=None, label_true=None, label_false=None)
  just like query_valid(shotlist=None, tag_true=None, tag_false=None)

- query_range(label_list, lower=None, upper=None, shotlist=None)
  这个有个小问题，这个智能查伦范围？能不能同时查询范围和诊断有效？

- count_label(shot_list, label_list, need_nd=False, show=True)
  just like db.tag_count(shotlist, taglist, needND=False, show=True)
  show 是啥？

## update labels

- put_label(shot_no,labels)
  给某一跑写入一些 label，label 是一个字典，key 是 label， value 是 label 的值
