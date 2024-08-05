 
import pandas as pd
import nltk 
def get_father_dir(splited_dir):
    from pathlib import Path
    path = Path(splited_dir)
    whole_dataset_dir=path.parent.absolute()
    return whole_dataset_dir
  
 