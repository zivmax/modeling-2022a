import pandas as pd

# Load the Excel file
xlsx_file = pd.ExcelFile('./data.xlsx')

# Iterate over each sheet in the Excel file
for sheet_name in xlsx_file.sheet_names:
    # Read the sheet as a DataFrame
    df = pd.read_excel(xlsx_file, sheet_name)
    
    # Save the DataFrame to a CSV file
    csv_file_name = f'{sheet_name}.csv'
    df.to_csv(csv_file_name, index=False)
