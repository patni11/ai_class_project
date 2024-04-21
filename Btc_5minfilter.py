import csv
import os
from datetime import datetime

# Get the list of CSV files in the current directory
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

# Initialize an empty list to store all filtered data
combined_data = []

for csv_file in csv_files:
    # Open the input CSV file
    with open(csv_file, 'r') as input_file:
        reader = csv.DictReader(input_file)
        data = list(reader)

    # Filter the data to include every 12-hour interval
    filtered_data = []
    for row in data:
        date_str, time_str = row['date'].split()
        month, day, year = [int(x) for x in date_str.split('/')]
        hour, minute = [int(x) for x in time_str.split(':')]
        date_time = datetime(year=2000 + int(year), month=month, day=day, hour=hour, minute=minute)
        if date_time.hour % 12 == 0:
            filtered_data.append(row)

    # Append the filtered data to the combined list
    combined_data.extend(filtered_data)

# Write the combined filtered data to a new CSV file
output_file = "combined_filtered_data.csv"
fieldnames = data[0].keys()
with open(output_file, 'w', newline='') as output_file:
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(combined_data)

print("Data filtered and combined.")