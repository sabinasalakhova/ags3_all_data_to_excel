

# 🔧 Setup Instructions
## Install dependencies
Make sure you have the following Python packages installed:
pip install pandas xlsxwriter
## Download the script folder
 Clone or download from Git the folder containing ags_3_reader.py.
## Run the script in terminal
 Navigate to the folder:
cd path/to/your/folder
## Then run the parser:
python ags_3_reader.py <input_folder_name> -o <output_filename>.xlsx

## - Example:
python ags_3_reader.py ags_data -o output.xlsx
## -  ⚠️ Make sure the output file name does not already exist in your AGS data folder.
### Check your output
The generated Excel file will appear in the input folder you specified.
If any file in the folder is in AGS4 format, the script will display a warning in the terminal.
