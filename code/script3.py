import csv

def count_universities(file_path):
  with open(file_path, mode='r', newline='') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    university_count = sum(1 for row in reader)
  return university_count

if __name__ == "__main__":
  file_path = 'universities2.csv'
  count = count_universities(file_path)
  print(f'The number of universities is: {count}')
  def count_unique_universities(file_path):
    with open(file_path, mode='r', newline='') as file:
      reader = csv.reader(file)
      next(reader)  # Skip the header row
      university_names = set(row[0] for row in reader)  # Assuming the university name is in the first column
    return len(university_names)

  if __name__ == "__main__":
    file_path = 'universities2.csv'
    unique_count = count_unique_universities(file_path)
    print(f'The number of unique universities is: {unique_count}')