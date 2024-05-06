import cv2
import socket
import struct
import pyzed.sl as sl
import math
import time
import datetime

def server_processing(image_data):
    try:
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', image_data)
        frame_data = buffer.tobytes()
        frame_size = len(frame_data)

        # Send frame size and frame data
        client_socket.sendall(struct.pack("!I", frame_size))
        client_socket.sendall(frame_data)

        # Receive output from the server
        response = client_socket.recv(1024)
        
        # Process the received response further if needed
        if response:
            return response
        else:
            return None
        
    except Exception as e:
        print("Error in server processing:", str(e))
        return None

def parse_results(response):
    # print("response to parse:",response)
    values = response.split()

    # Initialize empty list to store parsed values
    parsed_values = []

    # Iterate over the values in chunks of 6
    # print("reached to loop")
    for i in range(0, len(values), 6):
        # Convert the second value to float and rest to integers
        sublist = [int(values[i]), float(values[i+1])] + [int(x) for x in values[i+2:i+6]]
        # Append the sublist to parsed_values
        parsed_values.append(sublist)
    return parsed_values

# Initialize ZED camera
serial_number = 35633216
# serial_number = 36659489
# serial_number = 32088047
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
init_params.coordinate_units = sl.UNIT.METER
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
init_params.depth_maximum_distance = 50
init_params.set_from_serial_number(serial_number)

# Open ZED camera
status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED camera")
    exit(1)

left_image = sl.Mat()
point_cloud = sl.Mat()

# Connect to the server
server_address = ('192.168.49.2', 30020)
# server_address = ('192.168.20.245', 8080)
# server_address = ('192.168.20.156', 30010)
# server_address = ('192.168.21.176', 8080)
print('Connected to Cache Server.')
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(server_address)

try:
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)

            # Convert the image to a format suitable for YOLO
            image_data1 = left_image.get_data()
            image_data = image_data1[:,:,:3]
            image_sent = image_data[:,:,:3].copy()

            start_time = time.time()
            # Send frame size and data for processing
            detections_1 = server_processing(image_sent)
            # print("String Data:",detections_1)
            # print(type(detections_1))
            if detections_1 is None:
                print("No detections")
                continue  # Skip to the next iteration if detections_1 is None

            
            detections_2 = detections_1.decode('utf-8', 'ignore')
            # print("String Data after decoding:",detections_2)
            # Find the index of the "flag" substring
            start_index = detections_2.find("start")
            stop_index =detections_2.find("stop")
            if start_index != -1 and stop_index != -1:  # Ensure both "start" and "stop" are found
                start_index += len("start")  # Adjust to start after "start" substring
                detections = detections_2[start_index:stop_index].strip()
                # print("Substring between 'start' and 'stop':", detections)

            # Extract the substring after "flag"
            # detections = detections_2[index + len("flag"):].strip()
            
            # print("String Data after processing decoded string:",detections)

            # print("detection datatype:",type(detections))
            if start_index != -1 and detections != "":
                # Parse the results
                label_conf_bbox = parse_results(detections)
                # print("info list:",label_conf_bbox)
                for box in label_conf_bbox:
                    cls_type = box[0]
                    # print("BOX:",box)
                    confidence = box[1]
                    x1, y1, x2, y2 = box[2:]

                    # Calculate center of the bounding box
                    center_x = round((x1 + x2) / 2)
                    center_y = round((y1 + y2) / 2)

                    # Assuming you have some function to get point cloud value at the center of the bounding box
                    err, point_cloud_value = point_cloud.get_value(center_x, center_y)
                    center_3d_x = point_cloud_value[0]
                    center_3d_y = point_cloud_value[1]
                    center_3d_z = point_cloud_value[2]

                    # Calculate distance from the camera to the center of the bounding box
                    distance = round(math.sqrt(pow(center_3d_x, 2) + pow(center_3d_y, 2) + pow(center_3d_z, 2)), 2)

                    # Format the text to display
                    depth_1 = "{:.2f}".format(distance)
                    text = f'{cls_type}:{depth_1}m'

                    # Draw bounding box and display class type along with the calculated distance
                    color = (255, 255, 0)
                    cv2.putText(image_data1, text, (x1, y1 - 8), cv2.FONT_HERSHEY_COMPLEX, 1.5, color, 3)
                    cv2.rectangle(image_data1, (x1, y1), (x2, y2), color, 5)

                
            end_time = time.time()
            delay = end_time - start_time
            print('Delay time (client to server):', delay)
            cv2.imshow('Client Frame', image_data1)

            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            print('Frame sent by client at:', current_time)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

except Exception as e:
    print("Error:", str(e))
finally:
    zed.close()
    client_socket.close()
    cv2.destroyAllWindows()
