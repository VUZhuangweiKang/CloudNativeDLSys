import csv

def save_log_to_csv(log_name):
    with open(f"logs/{log_name}") as f:
        rows = f.readlines()[1:]
    rows = [row.strip().split(', ') for row in rows]
    rows = [[item.split(': ')[1] for item in row] for row in rows]

    # Remove units from the values
    for row in rows:
        row[0] = row[0].replace('GB', '')
        row[1] = row[1].replace('KB', '')
        row[2] = row[2].replace(' seconds', '')

    csv_log_file = log_name.replace('txt', 'csv')
    with open(f"logs/{csv_log_file}", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['total_size', 'file_size', 'total_time'])
        writer.writerows(rows)


if __name__ == "__main__":
    save_log_to_csv('log1.txt')
    save_log_to_csv('log2.txt')
    save_log_to_csv('log3.txt')
    save_log_to_csv('log4.txt')
    save_log_to_csv('log5.txt')
    save_log_to_csv('log6.txt')
