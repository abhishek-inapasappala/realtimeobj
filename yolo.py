import cv2
import torch
import numpy as np
from time import time

class ImprovedObjectDetection:
    """
    Improved version with better detection accuracy and reduced false positives.
    """
    
    def __init__(self):
        """
        Initializes the class with default values.
        """
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Confidence threshold (raised from 0.3 to 0.6)
        self.conf_threshold = 0.6
        
        # Non-maximum suppression threshold
        self.iou_threshold = 0.45

    def load_model(self):
        """
        Loads YOLOv5 model from PyTorch Hub with custom settings.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)  # Using medium model for better accuracy
        
        # Update model parameters for better detection
        model.conf = 0.6  # Confidence threshold
        model.iou = 0.45  # NMS IoU threshold
        model.agnostic = False  # Class-agnostic NMS
        model.multi_label = False  # Single label per box
        
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using YOLOv5 model.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        return labels, cord

    def class_to_label(self, x):
        """
        Returns the label for a given class ID.
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes the results and the frame, and plots the bounding boxes on the frame.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        
        for i in range(n):
            row = cord[i]
            if row[4] >= self.conf_threshold:  # Use class confidence threshold
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                
                # Different colors for different classes
                if self.class_to_label(labels[i]) == 'person':
                    bgr = (0, 255, 0)  # Green for person
                elif self.class_to_label(labels[i]) == 'cell phone':
                    bgr = (255, 0, 0)  # Blue for cell phone
                else:
                    bgr = (0, 0, 255)  # Red for other objects
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                label_text = f"{self.class_to_label(labels[i])} {row[4]:.2f}"
                cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        
        return frame

    def __call__(self):
        """
        This function is called when class is executed.
        """
        cap = cv2.VideoCapture(0)  # 0 for default webcam
        
        # Set camera resolution (higher can sometimes help)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            start_time = time()
            ret, frame = cap.read()
            if not ret:
                break
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB (YOLOv5 expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.score_frame(frame_rgb)
            frame = self.plot_boxes(results, frame)
            
            end_time = time()
            fps = 1 / (end_time - start_time)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            
            cv2.imshow('Improved Object Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Create a new object and execute
detector = ImprovedObjectDetection()
detector()