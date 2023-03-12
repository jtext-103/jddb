# %%

# In this program you are show how to build a data processing pipeline

# this pipeline will first extract tearing mode freq and amp as 2 two fewatures,
# the result is output to an new file repo,

# then to another person begin to work on the repo
# some signal is dropped the rest is resample to 1kHz, and normalized
# last disruption label is added as a new signal "disruption"

# this code relies on basic_processor.py, the tearing mode feature extractor is defined there.

# %% import jddb and processor

# %%

# init the input filerepo
# init the new output filerepo

# init shot set
# instanciate processors
# batch processing

# %%
# init the input filerepo
# drop signal
# resamlpe and normalization
# 这里我有个小问题，就是processor的设计，process的时候，结果存了硬盘吗？还是没有，需要调save吗？
