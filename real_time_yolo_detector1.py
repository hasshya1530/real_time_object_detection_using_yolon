import cv2
 import numpy as np
 import time

 def initialize_detector(weights_path, config_path, names_path):
     """Loads the YOLO network and class names."""
     net = cv2.dnn.readNet(weights_path, config_path)
     with open(names_path, "r") as f:
         class_labels = [line.strip() for line in f.readlines()]
     layer_names_all = net.getLayerNames()
     output_layer_indices = [layer_names_all[i[0] - 1] for i in net.getUnconnectedOutLayers()]
     num_classes = len(class_labels)
     detection_colors = np.random.uniform(0, 255, size=(num_classes, 3))
     return net, class_labels, output_layer_indices, detection_colors

 def detect_objects(image, network, output_layers_names, img_height, img_width):
     """Performs object detection on an image."""
     blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
     network.setInput(blob)
     outputs = network.forward(output_layers_names)
     return outputs

 def process_detections(outputs, confidence_threshold, img_height, img_width):
     """Processes the detection outputs to get bounding boxes, confidences, and class IDs."""
     boxes = []
     confidences = []
     class_ids = []
     for output in outputs:
         for detection in output:
             scores = detection[5:]
             class_id = np.argmax(scores)
             confidence = scores[class_id]
             if confidence > confidence_threshold:
                 center_x = int(detection[0] * img_width)
                 center_y = int(detection[1] * img_height)
                 width = int(detection[2] * img_width)
                 height = int(detection[3] * img_height)
                 x = int(center_x - width / 2)
                 y = int(center_y - height / 2)
                 boxes.append([x, y, width, height])
                 confidences.append(float(confidence))
                 class_ids.append(class_id)
     return boxes, confidences, class_ids

 def draw_detections(frame, boxes, confidences, class_ids, class_labels, colors):
     """Draws bounding boxes and labels on the frame."""
     indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # Adjusted thresholds slightly
     if indices is not None:
         for i in indices.flatten():
             x, y, w, h = boxes[i]
             label = class_labels[class_ids[i]]
             confidence = confidences[i]
             color = colors[class_ids[i]]
             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
             text = f"{label}: {confidence:.2f}"
             cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
     return frame

 def display_fps(frame, start_time, frame_count):
     """Calculates and displays the frames per second."""
     elapsed_time = time.time() - start_time
     fps = frame_count / elapsed_time
     fps_text = f"FPS: {fps:.2f}"
     cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # Different font and color
     return frame

 def main():
     """Main function to run the YOLO object detection."""
     weights_file = "weights/yolov3-tiny.weights"
     config_file = "cfg/yolov3-tiny.cfg"
     names_file = "coco.names"
     video_path = "uk.mp4"

     net, class_labels, output_layers, colors = initialize_detector(weights_file, config_file, names_file)

     cap = cv2.VideoCapture(video_path)
     if not cap.isOpened():
         print(f"Error: Could not open video at {video_path}")
         return

     start_time = time.time()
     frame_count = 0

     while True:
         ret, frame = cap.read()
         if not ret:
             break
         frame_count += 1
         height, width = frame.shape[:2]

         detections = detect_objects(frame, net, output_layers, height, width)
         detection_boxes, detection_confidences, detection_class_ids = process_detections(detections, 0.3, height, width) # Slightly higher threshold

         frame_with_detections = draw_detections(frame, detection_boxes, detection_confidences, detection_class_ids, class_labels, colors)
         frame_with_fps = display_fps(frame_with_detections, start_time, frame_count)

         cv2.imshow("Object Detection Feed", frame_with_fps) # Different window name

         if cv2.waitKey(1) & 0xFF == 27:
             break

     cap.release()
     cv2.destroyAllWindows()

 if __name__ == "__main__":
     main()
