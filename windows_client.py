import socket
import csv
import os

HOST = '0.0.0.0'
PORT = 5000


def create_writer(filename):
    file_exists = os.path.exists(filename)
    f = open(filename, mode='a', newline='')
    writer = csv.writer(f)
    # if not file_exists:
    #     writer.writerow(['Value'])  
    return f, writer

min_file, min_writer = create_writer('min.csv')
max_file, max_writer = create_writer('max.csv')
avg_file, avg_writer = create_writer('avg.csv')

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

print(f"Windows server listening on port {PORT}...")

conn, addr = server.accept()
print(f"Connected by {addr}")

try:
    while True:
        data = conn.recv(1024).decode()
        if not data:
            break

        print(f"Received: {data}")
        temp_min, temp_max, temp_avg = map(float, data.strip().split(','))

        
        min_writer.writerow([temp_min])
        max_writer.writerow([temp_max])
        avg_writer.writerow([temp_avg])

        
        min_file.flush()
        max_file.flush()
        avg_file.flush()

        conn.send("Data written".encode())

finally:
    min_file.close()
    max_file.close()
    avg_file.close()
    conn.close()
    server.close()