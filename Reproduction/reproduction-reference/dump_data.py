import pandas as pd
import os

base = r'D:\22 社会科学数据挖掘\AI-LLM 暴露度\人工智能-大语言模型技术”暴露指数数据'

# List all xlsx files
xlsx_files = [
    'exposure_base_minor_soc.xlsx',
    'exposure_base_soc_detail.xlsx',
    'exposure_base_zl_occu.xlsx',
    'exposure_by_year_minor_soc.xlsx',
    'exposure_by_year_soc_detail.xlsx',
    'exposure_by_year_zl_occu.xlsx',
]

for fname in xlsx_files:
    path = os.path.join(base, fname)
    if not os.path.exists(path):
        print(f'{fname}: NOT FOUND at {path}')
        continue
    df = pd.read_excel(path)
    print(f'\n===== {fname} =====')
    print(f'Shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    print(df.head(30).to_string())
    print(f'... (total {len(df)} rows)')
