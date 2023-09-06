#server2 and client2
import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator

import socket, cv2, pickle, struct
import imutils
import threading
import pyshine as ps
import cv2

def load_model():
    # model = YOLO("yolov8n.pt")
    model = YOLO("yolov8l.pt")
    # model.fuse()
    return model

def predict(model, frame):
    results = model(frame)
    return results

def plot_bboxes(results, frame, class_names_dict, box_annotator):
    xyxys = []
    confidences = []
    class_ids = []

    # Extract detection for person class
    for result in results[0]:
        class_id = result.boxes.cls.cpu().numpy().astype(int)

        if class_id == 0:
            xyxys.append(result.boxes.xyxy.cpu().numpy())
            confidences.append(result.boxes.conf.cpu().numpy())
            class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

    detections = Detections(xyxy=results[0].boxes.xyxy.cpu().numpy(),
                            confidence=results[0].boxes.conf.cpu().numpy(),
                            class_id=results[0].boxes.cls.cpu().numpy().astype(int))

    # Format custom labels
    labels = [f"{class_names_dict[class_id]} {confidence:0.2f}" for _, confidence, class_id, tracker_id in detections]

    # Annotate and display frame
    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    return frame

def run_object_detection(frame,capture_index):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using Device:", device)

    model = load_model()
    class_names_dict = model.model.names
    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=3, text_thickness=3, text_scale=1.5)

    cap = cv2.VideoCapture(capture_index)
    assert cap.isOpened()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        start_time = time()

        ret, frame = cap.read()
        assert ret

        results = predict(model, frame)
        frame = plot_bboxes(results, frame, class_names_dict, box_annotator)

        end_time = time()
        fps = 1 / np.round(end_time - start_time, 2)

        # cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5)

        cv2.imshow("YOLOv8 Detection", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Here the server is receiving video from the client
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
model = load_model()
class_names_dict = model.model.names
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=3, text_thickness=3, text_scale=1.5)
font_scale=3
font=cv2.FONT_HERSHEY_PLAIN

server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_name  = socket.gethostname()
host_ip = socket.gethostbyname(host_name)
print('HOST IP:',host_ip)
port = 9999
socket_address = (host_ip,port)
server_socket.bind(socket_address)
server_socket.listen()
print("Listening at",socket_address)

def show_client(addr,client_socket):
	try:
		print('CLIENT {} CONNECTED!'.format(addr))
		if client_socket: # if a client socket exists
			data = b""
			payload_size = struct.calcsize("Q")
			while True:
				while len(data) < payload_size:
					packet = client_socket.recv(4*1024) # 4K
					if not packet: break
					data+=packet
				packed_msg_size = data[:payload_size]
				data = data[payload_size:]
				msg_size = struct.unpack("Q",packed_msg_size)[0]
				
				while len(data) < msg_size:
					data += client_socket.recv(8*1024)
				frame_data = data[:msg_size]
				data  = data[msg_size:]
				frame = pickle.loads(frame_data)
				
				results = predict(model, frame)
				frame=plot_bboxes(results, frame, class_names_dict, box_annotator)
                
				#text  =  f"CLIENT: {addr}"
				#frame =  ps.putBText(frame,text,10,10,vspace=10,hspace=1,font_scale=0.7, background_RGB=(255,0,0),text_RGB=(255,250,250))
				cv2.imshow(f"FROM {addr}",frame)
				key = cv2.waitKey(1) & 0xFF
				if key  == ord('q'):
					break
			client_socket.close()
	except Exception as e:
		print(f"CLINET {addr} DISCONNECTED")
		pass
		
while True:
	client_socket,addr = server_socket.accept()
	thread = threading.Thread(target=show_client, args=(addr,client_socket))
	thread.start()
	print("TOTAL CLIENTS ",threading.activeCount() - 1)
	
				

