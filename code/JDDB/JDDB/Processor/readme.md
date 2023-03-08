# How to use Processor （明总）

和以前的设计一样，唯一的区别就是依赖 File Repo

ShotSet 不在直接对应文件夹和文件，而是使用 FileRepo 来找文件

输出也是一样，如果要输出到别的文件，就给一个输出的 FileRepo

构造函数 ShotSet(file_repo)

shot_set.process(.....,output_repo: Optional[FileRepo] = None)
