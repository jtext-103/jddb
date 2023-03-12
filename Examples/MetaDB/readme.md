# MetaDB Example

The mongodb dump file here is a MongoDB Collection dump of a MetaDB of J-TEXT.

This file is created wiht mongdump.....

```bash
# 老艾，你和大家商量一下确定dump出哪些炮，dump 20-30炮吧
mongodump --db your_database_name --collection your_collection_name --query '{ your_query }'
```

the query was to select a portial of J-TEXT shot.

It only have 20 shots.

To restore a MetaDB on your machine:

Install MongoDB (version>?) and then

use(下面是 mongorestor 的命令)，老艾你补全并测试一下

```bash

```

to restore the dumped data to you mongodb server
