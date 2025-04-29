import cv2
 import numpy as np
 import time

 def load_yolo_model(weights_path, config_path, names_path):
     """Loads the YOLOv3-tiny model and associated class names."""
     network = cv2.dnn.readNet(weights_path, config_path)
     with open(names_path, 'r') as file:
         class_names = [line.strip() for line in file.readlines()]
     layer_outputs_names = [network.getLayerNames()[i - 1] for i in network.getUnconnectedOutLayers()]
     num_classes = len(class_names)
     bounding_box_colors = np.random.uniform(0, 255, size=(num_classes, 3))
     return network, class_names, layer_outputs_names, bounding_box_colors

 def process_frame(frame, network, output_layers_names, confidence_threshold=0.2):
     """Performs object detection on a single frame."""
     height, width = frame.shape[:2]
     blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
     network.setInput(blob)
     layer_outputs = network.forward(output_layers_names)

     boxes = []
     confidences = []
     class_indices = []

     for output in layer_outputs:
         for detection in output:
             scores = detection[5:]
             class_id = np.argmax(scores)
             confidence = scores[class_id]
             if confidence > confidence_threshold:
                 center_x = int(detection[0] * width)
                 center_y = int(detection[1] * height)
                 box_width = int(detection[2] * width)
                 box_height = int(detection[3] * height)
                 x = int(center_x - box_width / 2)
                 y = int(center_y - box_height / 2)
                 boxes.append([x, y, box_width, box_height])
                 confidences.append(float(confidence))
                 class_indices.append(class_id)

     return boxes, confidences, class_indices, width, height

 def draw_detected_objects(frame, boxes, confidences, class_indices, class_names, colors, nms_threshold=0.3):
     """Draws bounding boxes and labels of detected objects on the frame."""
     indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, nms_threshold)
     if indices is not None:
         for i in indices.flatten():
             x, y, w, h = boxes[i]
             label = class_names[class_indices[i]]
             confidence = confidences[i]
             color = colors[class_indices[i]]
             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
             caption = f"{label}: {confidence:.2f}"
             cv2.putText(frame, caption, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
     return frame

 def display_performance(frame, start_time, frame_count):
     """Calculates and displays the frame rate on the frame."""
     elapsed_time = time.time() - start_time
     frames_per_second = frame_count / elapsed_time
     fps_text = f"Speed: {frames_per_second:.2f} FPS"
     cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
     return frame

 def main():
     """Main function to execute YOLOv3-tiny object detection on video."""
     weights_file = "weights/yolov3-tiny.weights"
     config_file = "cfg/yolov3-tiny.cfg"
     names_file = "coco.names"
     video_file = "usa-street.mp4"

     yolo_network, class_labels, output_layers_names, colors_for_boxes = load_yolo_model(weights_file, config_file, names_file)

     video_capture = cv2.VideoCapture(video_file)
     if not video_capture.isOpened():
         print(f"Error: Could not open video file: {video_file}")
         return

     start = time.time()
     frame_number = 0

     while True:
         retrieved, current_frame = video_capture.read()
         if not retrieved:
             break
         frame_number += 1

         detected_boxes, detection_scores, detected_classes, frame_width, frame_height = process_frame(current_frame, yolo_network, output_layers_names)

         frame_with_objects = draw_detected_objects(current_frame, detected_boxes, detection_scores, detected_classes, class_labels, colors_for_boxes)

         final_frame = display_performance(frame_with_objects, start, frame_number)

         cv2.imshow("Real-time Object Recognition", final_frame)

         if cv2.waitKey(1) & 0xFF == 27:
             break

     video_capture.release()
     cv2.destroyAllWindows()

 if __name__ == "__main__":
     main()
