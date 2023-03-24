# MetaDB Example

The mongodb dump file here is a MongoDB Collection dump of a MetaDB of J-TEXT.

This file is created wiht mongdump.....

```bash
mongodump --db your_database_name --collection your_collection_name --query '{ your_query }'
```

the query was to select a portial of J-TEXT shot.

It only have 20 shots.

To restore a MetaDB on your machine:

Install MongoDB (version>?) and then

use

```bash
mongorestore --host=localhost --port 27017 --db JDDB --collection Labels --dir Labels.bson
```

to restore the dumped data to you mongodb server
