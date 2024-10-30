import os
import csv

class CSVProtocol:
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = self.ensure_csv_extension(filename)
        self.file_path = os.path.join(self.folder, self.filename)
        self.create_folder()
        self.headers = self.define_headers()  # Define headers based on the filename

    def create_folder(self):
        # Create the folder if it does not exist
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    @staticmethod
    def ensure_csv_extension(filename):
        # Add .csv extension if it does not already have it
        if not filename.endswith('.csv'):
            filename += '.csv'
        return filename

    def define_headers(self):
        # Define headers based on the filename
        # Example logic: use the filename to create headers
        if self.filename.startswith('output'):
            return ['Column1', 'Column2', 'Column3']
        elif self.filename.startswith('data'):
            return ['FieldA', 'FieldB', 'FieldC']
        else:
            return ['Default1', 'Default2', 'Default3']

    def start_protocol(self):
        # Open the file in write mode (this will overwrite if it exists)
        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.headers)  # Write the headers as the first row

    def append_to_protocol(self, data):
        # Append data to the existing file
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)  # Write the data as a new row

# Example usage
folder = 'output'  # Specify your folder path
filename = 'output'              # Specify your filename without extension

# Create an instance of CSVProtocol
csv_protocol = CSVProtocol(folder, filename)

# Start the protocol by creating the file with predefined headers (overwrites if exists)
csv_protocol.start_protocol()

# Later, you can append data to the file
data_to_append = ['Data1', 'Data2', 'Data3']  # Example data to append
csv_protocol.append_to_protocol(data_to_append)
