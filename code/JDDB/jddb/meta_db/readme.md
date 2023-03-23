# How to use MetaDB （老艾）
## **ConnectDB**
Connect or disconnect to MetaDB 
   ```python
   from JDDB import meta_db
   c = meta_db.ConnectDB()
   ```


- ### **c.connect(connection_str, collection)**
  Description:   
  Connect to MetaDB befor any other action.  
  **Parameters: **  
  connection_string : Dictionary. The connections string to a mongodb server. Such as "host", "port", "username", "password" and so on.   
  collection : String. Collection name for the MetaDB.  
  Return:  
   Collection for the MetaDB
  
- ### **c.disconnect()**
  Description:   
  Disconnect from MetaDB after any other action.
  as name suggests.

  ### **Example :**
  ```python
  # Connect to the MetaDB
  connection_str = {
            "host" : "localhost",
            "port" : 27017,
            "username" : "DDBUser",
            "password" : "*******",
            "database": "JDDB"
          }
  collection = "Labels"
  c = meta_db.ConnectDB()
  labels = c.connect(connection_str,collection)

  # Disconnect from MetaDB
  c.disconnect()
  ```



## **MetaDB**
  Get meta or query eligible information from MetaDB.
 ```python
   from JDDB import meta_db
   c = meta_db.ConnectDB()
   labels = c.connect(connection_str,collection)
   db = meta_db.MetaDB(labels)                 # import MetaDB 
   ```

- ### **db.get_labels(shot_no)**  
  Description:  
  Get all meta of the shot inputed  
  Parameters:  
  shot_no : int or string. The shot number whose meta you want to get.  
  Return:  
  Dictionary. The meta you want.

  ### **Example :**
  ```python
  db.get_labels(1066648)

  -Return:
  {'shot': 1066648, 'ip': True, 'IsDisrupt': False, 'DownTime': 0.5923408076837159, 'bt': True, ... 'MA_TOR1_R09': True}
  ```

- ### **db.query(shot_list=None, filter=None)**
  Description:   
  Query the shots that meet the set conditions within the set shot number range.  
  Parameters:  
  shot_list : List. The range of shot numbers queried. If shot_list=None, query all shots in the MetaDB.  
  filter : Dictionary. The filter condition for the query. The description format of the condition must comply with Mongodb's specifications, and specific details can be found on the official website of Mongodb. If filter=None, Return all shot number in MetaDB.
  Return:  
  List. Shot number that meets the filter condition.  
  ### **Example :**
  ```python
  my_query = {'IsDisrupt': True, 'IpFlat':{'$gt':50}}
  db.query(my_query)

  -Return:
  [1046770, 1046794, 1046795, 1046806, 1046800 1046858, . . . , 1049184, 1050467, 1052286, 1050560, 1052295]
  ```


- ### **db.query_valid(shot_list=None, label_true=None, label_false=None)**
  Description:   
  For labels whose information stored in the database is True or False, return shot number that meets the filter condition.   
  Parameters:  
  shot_list : List. The range of shot numbers queried. If shot_list=None, query all shots in the MetaDB.  
  label_true : List of label names. Filter condition. The returned shots must satisfy that all labels in the label_true are True.  
  label_false : List of label names. Filter condition. The returned shots must satisfy that all labels in the label_false are False.
  Return:  
  List. Shot number that meets the filter condition.
  ### **Example :**  
  Get non-disruption shots with [" ip", " bt"] diagnostics available in the shot number range of [1064000, 1066649]
  ```python
  shot_list = [x for x in range(1064000, 1066649+1)]
  db.query(my_query)

  -Return:
  [1046770, 1046794, 1046795, 1046806, 1046800 1046858, . . . , 1049184, 1050467, 1052286, 1050560, 1052295]
  ```

- query_range(label_list, lower=None, upper=None, shot_list=None)
  这个有个小问题，这个智能查伦范围？能不能同时查询范围和诊断有效？  
  说明里面说一下可以级联query，，，，

- count_label(shot_list, label_list, need_nd=False, show=True)
  just like db.tag_count(shotlist, taglist, needND=False, show=True)
  show 是啥？

## update labels

- put_label(shot_no,labels)
  给某一跑写入一些 label，label 是一个字典，key 是 label， value 是 label 的值
