import cv2
import numpy as np
import threading
import socket
import struct
import time
import datetime
import math
from ultralytics import SAM

# Initialize the SAM model
model = SAM("sam_b.pt")
# for GPU
# model = SAM("sam_b.pt").cuda()

# Global variables for current frame and frame counts
frame_count_server = 0

def receive_frames(client_socket):
    global frame_count_server
    try:
        while True:
            # Receive frame size
            frame_size_data = client_socket.recv(4)
            if not frame_size_data:
                print("No frame size data received. Exiting.")
                break
            frame_size = struct.unpack("!I", frame_size_data)[0]

            # Receive frame data
            frame_data = b""
            while len(frame_data) < frame_size:
                packet = client_socket.recv(frame_size - len(frame_data))
                if not packet:
                    print("Incomplete frame data received. Exiting.")
                    break
                frame_data += packet

            if len(frame_data) != frame_size:
                print("Received frame size does not match expected size. Skipping frame.")
                continue

            # Decode the frame
            try:
                frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    print("Failed to decode frame. Skipping frame.")
                    continue
            except Exception as e:
                print("Error decoding frame:", e)
                continue

            print("Received frame:", frame.shape)

            # Process the frame with YOLO
            results = model.predict(frame, conf=0.5)

            # Send the results to the client
            # Create a string representation of results
            result_str2=''
            result_str = "start "
            for result in results:
                for obj in result.boxes:
                    x1, y1, x2, y2 = map(int, obj.xyxy[0])
                    label = int(obj.cls[0])
                    slabel=result.names[int(obj.cls[0])]
                    confidence = obj.conf[0]
                    result_str2 += f"Label: {slabel}, Confidence: {confidence}, Bounding Box: ({x1},{y1},{x2},{y2})\n"
                    result_str += f" {label} {confidence} {x1} {y1} {x2} {y2}"
            result_str+= " stop "
            print("Result:",result_str2)
            # print(len(result_str))

            # Send the results string
            if len(result_str)==12:
                print("No detections")
            client_socket.sendall(struct.pack("!I", len(result_str)))
            encoded_string = result_str.encode('utf-8')
            # print(encoded_string)
            client_socket.sendall(encoded_string)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display the frame
            # cv2.imshow("Server Frame", frame)
            cv2.waitKey(1)

            # Increment frame counts
            frame_count_server += 1
            print("Total frames processed:", frame_count_server)

    except Exception as e:
        print("Error receiving frame:", str(e))
    finally:
        cv2.destroyAllWindows()
        client_socket.close()


# Function to start the server
def start_server():
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to a host and port
    server_address = ('0.0.0.0', 8080)
    server_socket.bind(server_address)

    # Listen for incoming connections
    server_socket.listen(1)
    print("Server is listening for incoming connections...")

    # Accept incoming connections
    while True:
        client_socket, client_address = server_socket.accept()
        print("Connected to client:", client_address)

        # Start a new thread to handle the connection
        threading.Thread(target=receive_frames, args=(client_socket,)).start()

# Start the server
if __name__ == '__main__':
    start_server()
