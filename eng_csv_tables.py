import polars as pl
from pathlib import Path

# Define input and output directories
input_dir = Path("ratings/draws/word_draws")
all_names_dir = Path("data/all_names")
company_names_dir = Path("data/company_names")

# Create output directories if they don't exist
all_names_dir.mkdir(parents=True, exist_ok=True)
company_names_dir.mkdir(parents=True, exist_ok=True)

# English column mapping
column_mapping = {
    "word": "word",
    "association": "association",
    "wordtype": "word_type",
    "draw": "draw_number",
    "value": "rating_value"
}

# Mapping for filename translation
filename_mapping = {
    "bedrijfsnamen": "company_names",
    "namen": "names",
    "nepwoorden": "pseudowords",
    "betrouwbaar": "trustworthy",
    "slecht": "evil",
    "slim": "smart",
    "vrouwelijk": "feminine"
}

# Function to translate filenames to English
def translate_filename(dutch_name):
    parts = dutch_name.split("_")
    translated_parts = [filename_mapping.get(part, part) for part in parts]
    return "_".join(translated_parts)

# Dictionaries to hold dataframes for each association
dataframes_all_names = {}
dataframes_company_names = {}

# Loop over each parquet file and process it
for file in input_dir.glob("*.parquet"):
    print(f"Processing: {file.name}")

    # Read the parquet file
    df = pl.read_parquet(file)

    # Rename columns to English headers
    df = df.rename(column_mapping)

    # Translate filename to English
    new_file_stem = translate_filename(file.stem)

    # Extract association
    association = new_file_stem.split("_")[-1]

    # Collect DataFrames
    if association not in dataframes_all_names:
        dataframes_all_names[association] = []
    dataframes_all_names[association].append(df)

    if "company_names" in new_file_stem:
        if association not in dataframes_company_names:
            dataframes_company_names[association] = []
        dataframes_company_names[association].append(df)

# Save combined CSV files for each association
for association, dfs in dataframes_all_names.items():
    combined_df = pl.concat(dfs)
    combined_df.write_csv(all_names_dir / f"all_names_{association}.csv")
    print(f"✅ Saved combined file: {all_names_dir / f'all_names_{association}.csv'}")

for association, dfs in dataframes_company_names.items():
    combined_df = pl.concat(dfs)
    combined_df.write_csv(company_names_dir / f"company_names_{association}.csv")
    print(f"✅ Saved combined company file: {company_names_dir / f'company_names_{association}.csv'}")