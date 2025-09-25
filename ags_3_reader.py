import logging
from pathlib import Path
import argparse
import pandas as pd
from typing import Dict, List, Tuple, Optional
import re

# --------------------------------------------------------------------------------------
# Setup logging
# --------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Parsing and Data Handling Functions
# --------------------------------------------------------------------------------------

def _split_quoted_csv(s: str) -> List[str]:
    """Splits a CSV string, respecting quoted fields."""
    return [p.strip().strip('"') for p in re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', s)]

def parse_ags_file(file_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """
    Parses AGS 3.1 format file bytes into a dictionary of pandas DataFrames.
    """
    text = file_bytes.decode("latin-1", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    group_data: Dict[str, List[Dict[str, str]]] = {}
    current_group: Optional[str] = None
    headings: List[str] = []
    is_header_continuation = False

    def _split_line(line: str) -> List[str]:
        """split a CSV line"""
        # This regex handles commas inside quotes
        parts = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', line)
        # Strip quotes and whitespace from each part
        return [p.strip().strip('"') for p in parts]

    for line in lines:
        parts = _split_line(line)
        first_field = parts[0]

        # Rule 10: Start of a new GROUP (e.g., "**HOLE")
        if first_field.startswith("**"):
            current_group = first_field.strip("*")
            group_data.setdefault(current_group, [])
            headings = []
            is_header_continuation = False
            continue

        if not current_group:
            continue

        # Rule 11 & 13: HEADING line (e.g., "*HOLE_ID") or continuation
        if first_field.startswith("*") or is_header_continuation:
            # Strip '*' from the start of each heading
            new_headings = [h.lstrip("*") for h in parts]
            headings.extend(new_headings)
            # Check if the line ends with a comma, indicating continuation
            if line.endswith(","):
                is_header_continuation = True
            else:
                is_header_continuation = False
            continue

        # Rule 18: UNITS line - ignore for data parsing
        if first_field == "<UNITS>":
            continue

        # Rule 14: Data continuation line
        if first_field == "<CONT>":
            # This is a continuation of the previous data row.
            # The AGS spec is complex here. A simple approach is to merge,
            # but for now, we will skip to avoid data corruption.
            # A more advanced implementation would merge this with the last row.
            logger.warning(f"Data continuation line '<CONT>' found and skipped in group '{current_group}'.")
            continue

        # Data Row
        if headings and parts:
            # Ensure row has the same number of columns as headings
            row_values = parts[:len(headings)]
            # Pad row with empty strings if it's shorter than headings
            if len(row_values) < len(headings):
                row_values.extend([''] * (len(headings) - len(row_values)))
            
            row_dict = dict(zip(headings, row_values))
            group_data[current_group].append(row_dict)

    # Convert collected data into DataFrames
    group_dfs = {g: pd.DataFrame(rows) for g, rows in group_data.items() if rows}
    return group_dfs

def find_hole_id_column(columns: List[str]) -> Optional[str]:
    """Identify HOLE_ID or possible variants (just in case) in a list of columns."""
    # Create a map of uppercase column names to original names
    uc_map = {str(c).upper(): c for c in columns}
    
    # List of common primary keys for boreholes
    candidates = [
        "HOLE_ID", "HOLEID", "HOLE", "LOCA_ID", "LOCATION_ID", 
        "LOC_ID", "LOCID", "BORE_ID", "BOREID", "BOREHOLE_ID", "POINT_ID"
    ]
    
    for cand in candidates:
        if cand in uc_map:
            return uc_map[cand]
            
    return None

def process_files_to_combined_groups(input_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Processes all .ags files in a directory, combines them by group,
    and returns a single dictionary of combined DataFrames.
    """
    all_groups_data: Dict[str, List[pd.DataFrame]] = {}
    ags_files = sorted(list(input_dir.glob("*.ags")))

    if not ags_files:
        logger.warning(f"No .ags files found in {input_dir}")
        return {}

    logger.info(f"Found {len(ags_files)} AGS files to process.")

    for file_path in ags_files:
        logger.info(f"Processing file: {file_path.name}")
        try:
            file_bytes = file_path.read_bytes()
            groups = parse_ags_file(file_bytes)

            if not groups:
                logger.warning(f"No data groups parsed from {file_path.name}.")
                continue

            # Get prefix from filename (first 5 chars)
            filename_prefix = file_path.stem[:5]

            for group_name, df in groups.items():
                if df.empty:
                    continue
                
                df_copy = df.copy()
                
                # Add source file column
                df_copy["SOURCE_FILE"] = file_path.name

                # Find and prefix HOLE_ID
                hole_id_col = find_hole_id_column(df_copy.columns)
                if hole_id_col:
                    # Ensure column is string before modification
                    df_copy[hole_id_col] = df_copy[hole_id_col].astype(str).str.strip()
                    df_copy[hole_id_col] = f"{filename_prefix}_" + df_copy[hole_id_col]
                
                # Store the processed dataframe
                if group_name not in all_groups_data:
                    all_groups_data[group_name] = []
                all_groups_data[group_name].append(df_copy)

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    # Combine all dataframes for each group
    logger.info("Combining data from all files...")
    combined_dfs: Dict[str, pd.DataFrame] = {}
    for group_name, df_list in all_groups_data.items():
        if df_list:
            try:
                combined_dfs[group_name] = pd.concat(df_list, ignore_index=True)
            except Exception as e:
                logger.error(f"Could not combine data for group '{group_name}': {e}")
    
    return combined_dfs

def write_groups_to_excel(groups: Dict[str, pd.DataFrame], output_path: Path):
    """Writes a dictionary of DataFrames to a single Excel file with multiple sheets."""
    if not groups:
        logger.warning("No data to write to Excel.")
        return

    logger.info(f"Writing combined data to {output_path}...")
    try:
        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            for group_name, df in sorted(groups.items()):
                if df is None or df.empty:
                    continue
                
                # Sanitize sheet name: remove invalid chars and limit to 31 chars
                sanitized_name = re.sub(r'[\\/*?:\[\]]', '_', group_name)
                sheet_name = sanitized_name[:31]
                
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        logger.info("Successfully created Excel file.")
    except Exception as e:
        logger.error(f"Failed to write Excel file: {e}")

# --------------------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------------------

def main():
    """Main function to run the script from the command line."""
    parser = argparse.ArgumentParser(
        description="Batch process AGS files from a directory and combine them into a single Excel workbook."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the directory containing the .ags files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="AGS_Combined_Summary.xlsx",
        help="Name of the output Excel file. Defaults to 'AGS_Combined_Summary.xlsx'.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = input_path / args.output

    if not input_path.is_dir():
        logger.error(f"Error: Input directory not found at '{input_path}'")
        return

    # Process all files and get combined data
    combined_groups = process_files_to_combined_groups(input_path)

    # Write the combined data to an Excel file
    write_groups_to_excel(combined_groups, output_path)

if __name__ == "__main__":

    main()
