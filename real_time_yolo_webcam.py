import cv2
 import numpy as np

 def load_yolo_network(model_weights, model_config, class_names_file):
     """Initializes the YOLO network and loads class names."""
     neural_net = cv2.dnn.readNet(model_weights, model_config)
     with open(class_names_file, 'r') as f:
         object_classes = [line.strip() for line in f.readlines()]
     output_layer_names = [neural_net.getLayerNames()[i - 1] for i in neural_net.getUnconnectedOutLayers()]
     detection_colors = np.random.uniform(0, 255, size=(len(object_classes), 3))
     return neural_net, object_classes, output_layer_names, detection_colors

 def detect_objects_on_image(image, network, output_layers_names, confidence_threshold=0.5):
     """Performs object detection on the input image."""
     img_height, img_width = image.shape[:2]
     blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
     network.setInput(blob)
     predictions = network.forward(output_layers_names)

     bounding_boxes = []
     confidences_scores = []
     class_indices_detected = []

     for output in predictions:
         for detection in output:
             scores = detection[5:]
             class_index = np.argmax(scores)
             confidence = scores[class_index]
             if confidence > confidence_threshold:
                 center_x = int(detection[0] * img_width)
                 center_y = int(detection[1] * img_height)
                 box_width = int(detection[2] * img_width)
                 box_height = int(detection[3] * img_height)
                 x = int(center_x - box_width / 2)
                 y = int(center_y - box_height / 2)
                 bounding_boxes.append([x, y, box_width, box_height])
                 confidences_scores.append(float(confidence))
                 class_indices_detected.append(class_index)

     return bounding_boxes, confidences_scores, class_indices_detected

 def draw_labeled_boxes(image, boxes, confidences, class_indices, class_names, colors, nms_threshold=0.4):
     """Draws labeled bounding boxes on the image."""
     indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, nms_threshold)
     if indices is not None:
         font = cv2.FONT_HERSHEY_SIMPLEX
         for i in indices.flatten():
             x, y, w, h = boxes[i]
             label = class_names[class_indices[i]]
             color = colors[class_indices[i]]
             cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
             caption = f"{label}"
             cv2.putText(image, caption, (x, y + 30), font, 1, color, 2)
     return image

 def main():
     """Main function to perform object detection on an image using YOLOv3."""
     weights_file = "weights/yolov3.weights"
     config_file = "cfg/yolov3.cfg"
     names_file = "coco.names"
     image_file = "src_room.jpg"

     yolo_net, class_vocabulary, output_layer_names, box_colors = load_yolo_network(weights_file, config_file, names_file)

     input_image = cv2.imread(image_file)
     resized_image = cv2.resize(input_image, None, fx=0.8, fy=0.7)

     detected_boxes, detection_confidences, detected_classes = detect_objects_on_image(resized_image, yolo_net, output_layer_names)

     final_image = draw_labeled_boxes(resized_image, detected_boxes, detection_confidences, detected_classes, class_vocabulary, box_colors)

     print(cv2.dnn.NMSBoxes(detected_boxes, detection_confidences, 0.5, 0.4))
     cv2.imshow("Detected Objects", final_image)
     cv2.waitKey(0)
     cv2.destroyAllWindows()

 if __name__ == "__main__":
     main()
