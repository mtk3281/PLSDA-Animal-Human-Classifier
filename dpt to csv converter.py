import os
import csv

# Function to read .dpt file and extract wavelengths and intensities
def read_dpt_file(file_path):
    wavelengths = []
    intensities = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split('\t')
            wavelengths.append(float(parts[0]))
            intensities.append(float(parts[1]))
    return wavelengths, intensities

# Function to create the dataset
def create_dataset(input_folder, output_file):
    # Get all unique wavelengths across all files
    unique_wavelengths = set()
    for folder in ['animal', 'human']:
        for filename in os.listdir(os.path.join(input_folder, folder)):
            wavelengths, _ = read_dpt_file(os.path.join(input_folder, folder, filename))
            unique_wavelengths.update(wavelengths)

    # Sort unique wavelengths
    unique_wavelengths = sorted(unique_wavelengths)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['SAMPLE', 'CLASS', 'NAME'] + ['{}'.format(int(wavelength)) for wavelength in unique_wavelengths]
        writer.writerow(header)

        sample_count = 1

        # Processing animal samples
        for animal_file in os.listdir(os.path.join(input_folder, 'ANIMAL')):
            wavelengths, intensities = read_dpt_file(os.path.join(input_folder, 'ANIMAL', animal_file))
            animal_row = [sample_count, 0, 'ANIMAL']
            animal_intensity_dict = dict(zip(wavelengths, intensities))
            animal_row += [animal_intensity_dict.get(wavelength, 0) for wavelength in unique_wavelengths]
            writer.writerow(animal_row)
            sample_count += 1

        # Processing human samples
        for human_file in os.listdir(os.path.join(input_folder, 'HUMAN')):
            wavelengths, intensities = read_dpt_file(os.path.join(input_folder, 'HUMAN', human_file))
            human_row = [sample_count, 1, 'HUMAN']
            human_intensity_dict = dict(zip(wavelengths, intensities))
            human_row += [human_intensity_dict.get(wavelength, 0) for wavelength in unique_wavelengths]
            writer.writerow(human_row)
            sample_count += 1

# Path to the folder containing animal and human data
input_folder = 'full values/TRAIN/'

# Output CSV file
output_file = 'tempdataset.csv'

create_dataset(input_folder, output_file)
