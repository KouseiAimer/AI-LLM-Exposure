import pandas as pd
import os

# Find all xlsx files
base = r'D:\22 社会科学数据挖掘\AI-LLM 暴露度\人工智能-大语言模型技术"暴露指数数据'

files = [
    'exposure_base_minor_soc.xlsx',
    'exposure_base_soc_detail.xlsx',
    'exposure_base_zl_occu.xlsx',
    'exposure_by_year_minor_soc.xlsx',
    'exposure_by_year_soc_detail.xlsx',
    'exposure_by_year_zl_occu.xlsx',
]

output_lines = []

for fname in files:
    path = os.path.join(base, fname)
    df = pd.read_excel(path)

    output_lines.append(f'\n===== {fname} =====')
    output_lines.append(f'Shape: {df.shape}')
    output_lines.append(f'Columns: {list(df.columns)}')
    output_lines.append(f'Dtypes:\n{df.dtypes.to_string()}')

    # Convert to string
    output_lines.append('\n--- First 30 rows ---')
    output_lines.append(df.head(30).to_string())

    # Exposure values sorted
    if 'exposure' in df.columns:
        sorted_exp = sorted(df['exposure'].dropna().unique(), reverse=True)
        output_lines.append(f'\n--- Exposure values (high to low, {len(sorted_exp)} unique) ---')
        output_lines.append(str(sorted_exp[:50]))

# Write output
outpath = r'D:\22 社会科学数据挖掘\AI-LLM 暴露度\data_summary.txt'
with open(outpath, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print(f'Output written to {outpath}')
print(f'Total lines: {len(output_lines)}')
