import csv

# Open the input CSV file
with open('modified_data.csv', 'r') as input_file:
    reader = csv.DictReader(input_file)
    data = list(reader)

# Remove the 'unix' and 'symbol' columns
fieldnames = [fieldname for fieldname in data[0].keys() if fieldname not in ['Volume BTC']]
modified_data = [{fieldname: row[fieldname] for fieldname in fieldnames} for row in data]

# Write the modified data to a new CSV file
output_file = "btc_price_data.csv"
with open(output_file, 'w', newline='') as output_file:
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(modified_data)

print(f"Modified data written to '{output_file}'.")