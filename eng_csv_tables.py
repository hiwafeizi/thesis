import polars as pl
from pathlib import Path

# Define input and output directories
input_dir = Path("ratings/draws/word_draws")
output_dir = Path("data/processed")

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

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

# Loop over each parquet file and process it
for file in input_dir.glob("*.parquet"):
    print(f"Processing: {file.name}")

    # Read the parquet file
    df = pl.read_parquet(file)

    # Rename columns to English headers
    df = df.rename(column_mapping)

    # Translate filename to English
    new_file_stem = translate_filename(file.stem)

    # Save as CSV to the output directory
    csv_filename = output_dir / f"{new_file_stem}.csv"
    df.write_csv(csv_filename)

    print(f"✅ Saved to: {csv_filename}")
