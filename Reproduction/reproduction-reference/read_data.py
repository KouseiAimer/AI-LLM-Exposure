import pandas as pd
import json

base_dir = r'D:\22 社会科学数据挖掘\AI-LLM 暴露度'

# Try glob to find all xlsx files
import glob
files = glob.glob(base_dir + '/**/*.xlsx', recursive=True)
for f in files:
    print(f'=== {f} ===')
    df = pd.read_excel(f)
    print(f'Shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    print(df.head(20).to_string())
    print()
