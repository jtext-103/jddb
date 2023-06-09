# FileRepo Example

This is a FileRepo from J-TEXT.

It only have 20 shots, with limited signals. You can play with it to get familiar with J-TEXT data.

It was created by mds_dumper module. All the labels for those shot is stored in MetaDB in the example files. Be noted that this means the shot files does NOT have labels. However, this is intended, so you can try to restore a MetaDB and sync the labels to the FileRepo.

If you are going to label the shot, the best practice is to add labels to the MetaDB nit to shot files, and after the labeling work, sync to the FileRepo.

## Where can you find the files

The files is stored in Dropbox, here is the url:

https://www.dropbox.com/s/gge7amz1czt4l2d/Example_shot_files.zip?dl=0

After download, create a folder named `TestShots` here, then unzip the downloaded file into the `TestShots` folder for the examples to run.

The folder structure is like:

```
+---FileRepo
|   |
|   +---TestShots
|       +---10526XX
|       |   \---105263X
|       |           1052637.hdf5
|       |
|       +---10531XX
|       |   +---105310X
|       |   |       1053103.hdf5
|       |   |
|       |   \---105319X
|       |           1053198.hdf5
|       |
|       +---10537XX
|       |   \---105375X
|       |           1053759.hdf5
|       |
```

You can create you file repo with a base path of: `./FileRepo/$shot_2$xx/$shot_1$x/`
