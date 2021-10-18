import torch
import PIL as pl
import numpy as np
import sklearn as skl

if __name__=="__main__":
    print(torch.__version__,
          pl.__version__,
          skl.__version__,
          np.__version__,
          sep='\n')

