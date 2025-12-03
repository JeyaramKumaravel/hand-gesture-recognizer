import os 
import cv2 
import matplotlib.pyplot as plt 
import numpy as np
from tkinter import *
import tkinter as tk
from tkinter import ttk, filedialog
import glob
from gtts import gTTS
import pygame
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageTk
import joblib  # Changed this line - removed sklearn.externals
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight

class SignLanguageDetector:
    def __init__(self):
        # Initialize pygame mixer
        pygame.mixer.init()
        self.setup_gui()
        self.verify_dataset()
        self.setup_classifiers()
        # Add flag to track if GUI is active
        self.is_active = True
        
    def setup_gui(self):
        self.root = Tk()
        self.root.geometry("1200x800")  # Increased window size
        self.root.config(background="#2C3E50")  # Dark blue background
        self.root.title("Sign Language Detection Portal")

        # Create main container
        main_container = Frame(self.root, bg="#2C3E50")
        main_container.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Title Frame
        title_frame = Frame(main_container, bg="#2C3E50")
        title_frame.pack(fill=X, pady=(0, 10))

        title_label = Label(
            title_frame, 
            text="SIGN LANGUAGE DETECTION", 
            bg="#2C3E50",
            fg="white",
            font=("Helvetica", 24, "bold")
        )
        title_label.pack()

        # Create two main columns
        left_panel = Frame(main_container, bg="#34495E", padx=10, pady=10)
        right_panel = Frame(main_container, bg="#34495E", padx=10, pady=10)
        left_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=5)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True, padx=5)

        # Left Panel - Controls
        control_frame = Frame(left_panel, bg="#34495E")
        control_frame.pack(fill=X, pady=(0, 10))

        # Button styles
        button_style = {
            'font': ('Helvetica', 10),
            'width': 15,
            'height': 2,
            'bd': 0,
            'relief': 'flat',
            'cursor': 'hand2'
        }

        # Buttons with modern colors
        self.camera_btn = Button(
            control_frame,
            text="Use Camera",
            command=self.start_camera,
            bg="#3498DB",  # Blue
            fg="white",
            activebackground="#2980B9",
            **button_style
        )
        self.camera_btn.pack(side=LEFT, padx=5)

        self.browse_btn = Button(
            control_frame,
            text="Browse Image",
            command=self.browse_files,
            bg="#2ECC71",  # Green
            fg="white",
            activebackground="#27AE60",
            **button_style
        )
        self.browse_btn.pack(side=LEFT, padx=5)

        self.delete_btn = Button(
            control_frame,
            text="Reset Model",
            command=self.delete_models,
            bg="#E74C3C",  # Red
            fg="white",
            activebackground="#C0392B",
            **button_style
        )
        self.delete_btn.pack(side=LEFT, padx=5)

        self.exit_btn = Button(
            control_frame,
            text="Exit",
            command=self.root.destroy,
            bg="#95A5A6",  # Gray
            fg="white",
            activebackground="#7F8C8D",
            **button_style
        )
        self.exit_btn.pack(side=LEFT, padx=5)

        # Text Area with modern styling
        text_frame = Frame(left_panel, bg="#34495E")
        text_frame.pack(fill=BOTH, expand=True)

        text_label = Label(
            text_frame,
            text="Detection Results",
            bg="#34495E",
            fg="white",
            font=("Helvetica", 12, "bold")
        )
        text_label.pack(pady=(0, 5))

        self.text_area = Text(
            text_frame,
            height=15,
            width=40,
            bg="#2C3E50",
            fg="#E74C3C",  # Red text
            font=("Consolas", 11),
            padx=10,
            pady=10
        )
        self.text_area.pack(fill=BOTH, expand=True)

        # Right Panel - Image Display with Original Image larger
        self.image_labels = {}
        
        # Create grid for images with different layout
        grid_frame = Frame(right_panel, bg="#34495E")
        grid_frame.pack(fill=BOTH, expand=True)
        
        # Configure grid weights
        grid_frame.grid_columnconfigure(0, weight=1)  # Full width for original image
        grid_frame.grid_rowconfigure(0, weight=2)     # More height for original image
        grid_frame.grid_rowconfigure(1, weight=1)     # Less height for processing steps
        
        # Original Image - Full width
        orig_frame = Frame(grid_frame, bg="#34495E", padx=5, pady=5)
        orig_frame.grid(row=0, column=0, columnspan=3, sticky='nsew', padx=5, pady=5)
        
        orig_label = Label(
            orig_frame,
            text="Original Image / Camera Feed",
            bg="#34495E",
            fg="white",
            font=("Helvetica", 14, "bold")
        )
        orig_label.pack(pady=(0, 5))
        
        # Larger frame for original image
        orig_img_frame = Frame(orig_frame, bg="#2C3E50", padx=2, pady=2)
        orig_img_frame.pack(fill=BOTH, expand=True)
        
        orig_img_label = Label(
            orig_img_frame,
            bg="#2C3E50",
            width=800,    # Increased width
            height=400    # Maintained height
        )
        orig_img_label.pack(fill=BOTH, expand=True)
        self.image_labels['Original Image'] = orig_img_label
        
        # Processing steps in bottom row - 3 columns
        process_frame = Frame(grid_frame, bg="#34495E")
        process_frame.grid(row=1, column=0, columnspan=3, sticky='nsew')
        
        # Configure processing steps grid
        process_frame.grid_columnconfigure(0, weight=1)
        process_frame.grid_columnconfigure(1, weight=1)
        process_frame.grid_columnconfigure(2, weight=1)
        
        # Other processing steps - Smaller size in bottom row
        other_titles = ['Grayscale', 'Threshold', 'Contours']  # Reduced to 3 key steps
        
        for i, title in enumerate(other_titles):
            frame = Frame(process_frame, bg="#34495E", padx=5, pady=5)
            frame.grid(row=0, column=i, sticky='nsew', padx=5, pady=5)
            
            label = Label(
                frame,
                text=title,
                bg="#34495E",
                fg="white",
                font=("Helvetica", 10, "bold")
            )
            label.pack(pady=(0, 5))
            
            # Smaller frames for processing steps
            img_frame = Frame(frame, bg="#2C3E50", padx=2, pady=2)
            img_frame.pack(fill=BOTH, expand=True)
            
            img_label = Label(
                img_frame,
                bg="#2C3E50",
                width=250,     # Width for processing steps
                height=150     # Height for processing steps
            )
            img_label.pack(fill=BOTH, expand=True)
            
            self.image_labels[title] = img_label

        # Add hover effects to buttons
        for btn in [self.camera_btn, self.browse_btn, self.delete_btn, self.exit_btn]:
            btn.bind('<Enter>', lambda e, b=btn: self._on_enter(b))
            btn.bind('<Leave>', lambda e, b=btn: self._on_leave(b))

        # Camera state initialization
        self.camera_active = False
        self.cap = None
        
        # Progress window attributes
        self.progress_window = None
        self.progress_bar = None
        self.progress_label = None

    def _on_enter(self, button):
        """Button hover effect - brighten"""
        current_bg = button.cget('background')
        # Brighten the color slightly
        button.configure(bg=self._adjust_color(current_bg, 1.1))

    def _on_leave(self, button):
        """Button hover effect - restore"""
        current_bg = button.cget('background')
        # Restore original color
        button.configure(bg=self._adjust_color(current_bg, 0.9))

    def _adjust_color(self, color, factor):
        """Adjust color brightness"""
        # Convert color to RGB
        rgb = self.root.winfo_rgb(color)
        # Adjust each component
        new_rgb = tuple(min(65535, int(c * factor)) for c in rgb)
        # Convert back to color string
        return f'#{new_rgb[0]//256:02x}{new_rgb[1]//256:02x}{new_rgb[2]//256:02x}'

    def setup_classifiers(self):
        """Initialize and train classifier"""
        self.classifier = SGDClassifier(
            loss='modified_huber',  # Changed from 'hinge' for better probability estimates
            learning_rate='adaptive',
            eta0=0.01,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            n_jobs=-1,
            early_stopping=False,
            class_weight='balanced'
        )
        self.label_encoder = LabelEncoder()
        
        # Try to load pre-trained model
        if self.load_model():
            print("Loaded pre-trained model successfully!")
        else:
            print("Training new model...")
            self.train_classifier()
        
    def save_model(self):
        """Save trained model and label encoder to files"""
        try:
            # Create models directory if it doesn't exist
            if not os.path.exists('models'):
                os.makedirs('models')
                
            # Save the trained model
            joblib.dump(self.classifier, 'models/sign_language_model.joblib')
            # Save the label encoder
            joblib.dump(self.label_encoder, 'models/label_encoder.joblib')
            print("\nModel saved successfully!")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
        
    def load_model(self):
        """Load trained model and label encoder from files"""
        try:
            if os.path.exists('models/sign_language_model.joblib') and \
               os.path.exists('models/label_encoder.joblib'):
                # Load the trained model
                self.classifier = joblib.load('models/sign_language_model.joblib')
                # Load the label encoder
                self.label_encoder = joblib.load('models/label_encoder.joblib')
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
        
    def train_classifier(self):
        """Train classifier using mini-batches with GUI updates"""
        self.show_progress_window("Training Model")
        
        try:
            batch_size = 100
            features = []
            labels = []
            
            if not os.path.exists('dataset'):
                self.progress_label["text"] = "Error: 'dataset' folder not found!"
                return
                
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            
            # Count total files first
            self.update_progress(0, "Scanning dataset...")
            found_files = 0
            for sign_folder in os.listdir('dataset'):
                folder_path = os.path.join('dataset', sign_folder)
                if os.path.isdir(folder_path):
                    for filename in os.listdir(folder_path):
                        if os.path.splitext(filename)[1].lower() in valid_extensions:
                            found_files += 1
                self.root.update()  # Keep GUI responsive
            
            if found_files == 0:
                self.update_progress(100, "No valid files found!")
                return
                
            # Process files
            loaded_files = 0
            current_batch_features = []
            current_batch_labels = []
            
            for sign_folder in os.listdir('dataset'):
                folder_path = os.path.join('dataset', sign_folder)
                if not os.path.isdir(folder_path):
                    continue
                    
                for filename in os.listdir(folder_path):
                    if os.path.splitext(filename)[1].lower() not in valid_extensions:
                        continue
                        
                    try:
                        file_path = os.path.join(folder_path, filename)
                        img = cv2.imread(file_path)
                        if img is None:
                            continue
                            
                        feature = self.extract_features(img)
                        if feature is not None:
                            current_batch_features.append(feature)
                            current_batch_labels.append(sign_folder)
                            loaded_files += 1
                            
                            # Process batch if it reaches batch_size
                            if len(current_batch_features) >= batch_size:
                                features.extend(current_batch_features)
                                labels.extend(current_batch_labels)
                                current_batch_features = []
                                current_batch_labels = []
                        
                        # Update progress
                        progress = (loaded_files / found_files) * 100
                        self.update_progress(
                            progress,
                            f"Processing images: {loaded_files}/{found_files}"
                        )
                        self.root.update()  # Keep GUI responsive
                        
                    except Exception as e:
                        print(f"Error processing '{file_path}': {str(e)}")
            
            # Add remaining batch
            if current_batch_features:
                features.extend(current_batch_features)
                labels.extend(current_batch_labels)
            
            # Train model
            self.update_progress(0, "Training model...")
            
            y = self.label_encoder.fit_transform(labels)
            X = np.array(features, dtype=np.float32)
            
            # Compute class weights
            classes = np.unique(y)
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            class_weight_dict = dict(zip(classes, class_weights))
            
            # Initialize classifier
            self.classifier = SGDClassifier(
                loss='modified_huber',
                learning_rate='adaptive',
                eta0=0.01,
                max_iter=1000,
                tol=1e-3,
                random_state=42,
                n_jobs=-1,
                early_stopping=False,
                class_weight=class_weight_dict
            )
            
            # Train in mini-batches
            n_samples = len(X)
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            # First batch to initialize
            first_batch_end = min(batch_size, n_samples)
            self.classifier.partial_fit(
                X[:first_batch_end],
                y[:first_batch_end],
                classes=classes
            )
            
            # Remaining batches
            for i in range(batch_size, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                self.classifier.partial_fit(
                    X[i:batch_end],
                    y[i:batch_end]
                )
                
                # Update progress
                progress = (i / n_samples) * 100
                self.update_progress(
                    progress,
                    f"Training model: {i}/{n_samples} samples"
                )
                self.root.update()
            
            self.update_progress(100, "Saving model...")
            self.save_model()
            
            self.update_progress(100, "Training completed!")
            
        except Exception as e:
            self.update_progress(100, f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Close progress window after a short delay
            self.root.after(1500, self.progress_window.destroy)

    def extract_features(self, img):
        """Extract features with improved parameters for better accuracy"""
        if img is None:
            return None
            
        try:
            # Resize image consistently
            img = cv2.resize(img, (128, 128))  # Increased size for more detail
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Denoise image
            gray = cv2.fastNlMeansDenoising(gray)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )
            
            # Extract HOG features with optimized parameters
            winSize = (128,128)
            blockSize = (16,16)
            blockStride = (8,8)
            cellSize = (8,8)
            nbins = 9
            hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
            hog_features = hog.compute(thresh)
            
            # Extract LBP features
            radius = 3
            n_points = 8 * radius
            lbp = self.local_binary_pattern(gray, n_points, radius)
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            lbp_hist = lbp_hist.astype(np.float32)
            lbp_hist /= (lbp_hist.sum() + 1e-7)
            
            # Extract color features from multiple color spaces
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Calculate histograms for each channel
            hist_features = []
            for color_img in [img_hsv, img_lab]:
                for i in range(3):
                    hist = cv2.calcHist([color_img], [i], None, [32], [0, 256])
                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                    hist_features.append(hist.flatten())
            
            # Combine all features
            color_features = np.concatenate(hist_features)
            hog_features = hog_features.flatten()
            
            # Normalize HOG features
            hog_features = (hog_features - np.mean(hog_features)) / (np.std(hog_features) + 1e-7)
            
            # Combine all features
            features = np.concatenate([hog_features, lbp_hist, color_features]).astype(np.float32)
            
            # Handle NaN and inf values
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None

    def local_binary_pattern(self, image, n_points, radius):
        """Compute local binary pattern features"""
        points = np.zeros((n_points, 2))
        for i in range(n_points):
            theta = float(i * 2 * np.pi) / n_points
            points[i] = [-radius * np.sin(theta), radius * np.cos(theta)]
            
        shape = image.shape
        output = np.zeros(shape, dtype=np.uint8)
        
        for i in range(radius, shape[0] - radius):
            for j in range(radius, shape[1] - radius):
                value = 0
                center = image[i, j]
                for k, point in enumerate(points):
                    y, x = point
                    y = int(round(i + y))
                    x = int(round(j + x))
                    value |= (image[y, x] >= center) << k
                output[i, j] = value
                
        return output

    def predict_sign(self, img):
        """Predict sign using classifier with improved confidence handling"""
        feature = self.extract_features(img)
        if feature is None:
            return None, 0.0
        
        try:
            # Get decision scores
            decisions = self.classifier.decision_function([feature])
            
            # Convert to probabilities using calibrated softmax
            probs = self._calibrated_softmax(decisions)
            
            # Get top predictions
            top_idx = np.argsort(probs[0])[-3:][::-1]
            top_prob = probs[0][top_idx]
            top_signs = self.label_encoder.inverse_transform(top_idx)
            
            # Apply confidence thresholding
            if top_prob[0] < 0.4:  # Increased confidence threshold
                return None, 0.0
                
            # Check for ambiguous predictions
            if top_prob[0] - top_prob[1] < 0.2:  # Check if top predictions are too close
                return None, 0.0
            
            return top_signs[0], top_prob[0]
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None, 0.0

    def _calibrated_softmax(self, x):
        """Improved softmax with temperature scaling"""
        temperature = 1.5  # Adjust this value to control prediction sharpness
        x = x / temperature
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def browse_files(self):
        """Browse and load image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=(
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            )
        )
        
        if file_path:
            try:
                # Read and process image
                img = cv2.imread(file_path)
                if img is None:
                    self.text_area.insert(tk.END, "Error: Could not read image file\n")
                    return
                    
                # Get prediction
                predicted_sign, confidence = self.predict_sign(img)
                
                # Show preprocessing steps
                self.show_preprocessing_steps(img)
                
                # Update prediction text
                if predicted_sign:
                    self.text_area.delete(1.0, tk.END)
                    result_text = f"Predicted Sign: {predicted_sign}\n"
                    result_text += f"Confidence: {confidence:.2f}\n"
                    self.text_area.insert(tk.END, result_text)
                    
            except Exception as e:
                self.text_area.insert(tk.END, f"Error processing image: {str(e)}\n")

    def show_preprocessing_steps(self, img):
        """Display image processing steps"""
        try:
            def prepare_image(img):
                """Helper function to prepare image for display"""
                if len(img.shape) == 2:  # Grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 3:  # Color
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img

            def resize_and_pad(img, target_size):
                """Resize image maintaining aspect ratio and pad if necessary"""
                h, w = img.shape[:2]
                target_w, target_h = target_size
                
                # Calculate scaling factor
                scale = min(target_w/w, target_h/h)
                new_w, new_h = int(w*scale), int(h*scale)
                
                # Resize image
                resized = cv2.resize(img, (new_w, new_h))
                
                # Create black padding
                padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                
                # Calculate padding
                x_offset = (target_w - new_w) // 2
                y_offset = (target_h - new_h) // 2
                
                # Place resized image on padded background
                padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                
                return padded

            # 1. Original Image
            original = prepare_image(img.copy())
            original = resize_and_pad(original, (800, 400))  # Larger size for original
            original_img = ImageTk.PhotoImage(Image.fromarray(original))
            self.image_labels['Original Image'].configure(image=original_img)
            self.image_labels['Original Image'].image = original_img
            
            # 2. Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_img = resize_and_pad(prepare_image(gray), (250, 150))
            gray_tk = ImageTk.PhotoImage(Image.fromarray(gray_img))
            self.image_labels['Grayscale'].configure(image=gray_tk)
            self.image_labels['Grayscale'].image = gray_tk
            
            # 3. Threshold
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            thresh_img = resize_and_pad(prepare_image(thresh), (250, 150))
            thresh_tk = ImageTk.PhotoImage(Image.fromarray(thresh_img))
            self.image_labels['Threshold'].configure(image=thresh_tk)
            self.image_labels['Threshold'].image = thresh_tk
            
            # 4. Contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_img = img.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
            contour_img = resize_and_pad(prepare_image(contour_img), (250, 150))
            contour_tk = ImageTk.PhotoImage(Image.fromarray(contour_img))
            self.image_labels['Contours'].configure(image=contour_tk)
            self.image_labels['Contours'].image = contour_tk
            
        except Exception as e:
            print(f"Error showing preprocessing steps: {str(e)}")

    def run(self):
        """Start the application"""
        try:
            self.root.mainloop()
        finally:
            self.is_active = False  # Mark GUI as inactive when window closes
            self.stop_camera()

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.is_active = False  # Mark GUI as inactive
            self.stop_camera()
            pygame.mixer.quit()
        except:
            pass  # Ignore any errors during cleanup

    def verify_dataset(self):
        """Verify dataset folder structure and provide guidance"""
        try:
            if not os.path.exists('dataset'):
                print("\nAttempting to create 'dataset' folder...")
                os.makedirs('dataset')
                print("Successfully created 'dataset' folder")
        except PermissionError:
            print("\nERROR: Cannot create 'dataset' folder due to permission denied.")
            print("Please try one of these solutions:")
            print("1. Run the program as administrator")
            print("2. Manually create a folder named 'dataset' in:")
            print(f"   {os.path.abspath(os.path.curdir)}")
            print("3. Change the working directory to a location where you have write permissions")
            
        if not os.path.exists('dataset'):
            print("\nWARNING: No 'dataset' folder found!")
            print("The program will continue, but you need to create the dataset folder")
            print("and add sign language images before training can begin.\n")
            
        print("\nDataset Requirements:")
        print("1. Organize sign images in subdirectories by sign")
        print("2. Name folders according to the signs (e.g., 'A', 'B', 'Hello')")
        print("3. Include multiple examples in each folder")
        print("4. Supported formats: .jpg, .jpeg, .png, .bmp")
        print("\nExample dataset structure:")
        print("dataset/")
        print("├── A/")
        print("│   ├── A1.jpg")
        print("│   ├── A2.jpg")
        print("│   └── A3.jpg")
        print("├── B/")
        print("│   ├── B1.jpg")
        print("│   ├── B2.jpg")
        print("│   └── B3.jpg")
        print("└── Hello/")
        print("    ├── Hello1.jpg")
        print("    ├── Hello2.jpg")
        print("    └── Hello3.jpg\n")

    def start_camera(self):
        """Toggle camera on/off"""
        if not self.camera_active:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    self.text_area.insert(tk.END, "Error: Could not open camera\n")
                    return
                    
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                self.camera_active = True
                self.camera_btn.config(text="Stop Camera")
                self.text_area.delete(1.0, tk.END)
                self.text_area.insert(tk.END, "Camera started. Place hand in green box.\n")
                
                # Start processing frames
                self.process_camera_feed()
                
            except Exception as e:
                self.text_area.insert(tk.END, f"Error starting camera: {str(e)}\n")
        else:
            self.stop_camera()

    def stop_camera(self):
        """Stop the camera feed"""
        self.camera_active = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_btn.config(text="Use Camera")
        self.text_area.insert(tk.END, "Camera stopped\n")

    def process_camera_feed(self):
        """Process camera feed with optimized performance"""
        if not self.camera_active:
            return
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.text_area.insert(tk.END, "Error: Cannot read from camera\n")
                self.stop_camera()
                return
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Add corner guide rectangle
            height, width = frame.shape[:2]
            guide_size = min(width, height) // 2
            guide_x = width - guide_size - 20
            guide_y = 20
            
            # Draw guide box
            cv2.rectangle(frame, 
                         (guide_x, guide_y), 
                         (guide_x + guide_size, guide_y + guide_size),
                         (0, 255, 0), 3)
            
            # Process only the guide region
            roi = frame[guide_y:guide_y + guide_size, 
                       guide_x:guide_x + guide_size]
            
            # Get prediction
            predicted_sign, confidence = self.predict_sign(roi)
            
            # Show preprocessing steps
            self.show_preprocessing_steps(roi)
            
            # Show the frame with guide - Resize for better display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (500, 400))  # Resize to match label size
            img = Image.fromarray(frame_rgb)
            img = ImageTk.PhotoImage(image=img)
            self.image_labels['Original Image'].configure(image=img)
            self.image_labels['Original Image'].image = img
            
            # Update prediction text
            if predicted_sign and confidence > 0.4:
                self.text_area.delete(1.0, tk.END)
                result_text = f"Predicted Sign: {predicted_sign}\n"
                result_text += f"Confidence: {confidence:.2f}\n"
                
                if confidence > 0.7:
                    result_text += "Status: High confidence prediction\n"
                elif confidence > 0.5:
                    result_text += "Status: Medium confidence prediction\n"
                else:
                    result_text += "Status: Low confidence prediction\n"
                    
                self.text_area.insert(tk.END, result_text)
            
            # Schedule next frame
            if self.camera_active:
                self.root.after(10, self.process_camera_feed)
                
        except Exception as e:
            self.text_area.insert(tk.END, f"Camera error: {str(e)}\n")
            self.stop_camera()

    def delete_models(self):
        """Delete the models folder and retrain"""
        try:
            if os.path.exists('models'):
                import shutil
                shutil.rmtree('models')
                self.text_area.insert(tk.END, "Models deleted successfully.\n")
                
                # Retrain the model
                self.text_area.insert(tk.END, "Retraining model...\n")
                self.text_area.update()  # Update GUI to show message
                self.setup_classifiers()
                self.text_area.insert(tk.END, "Retraining complete.\n")
            else:
                self.text_area.insert(tk.END, "No models folder found.\n")
        except Exception as e:
            self.text_area.insert(tk.END, f"Error deleting models: {str(e)}\n")

    def show_progress_window(self, title):
        """Create and show a progress window"""
        self.progress_window = Toplevel(self.root)
        self.progress_window.title(title)
        self.progress_window.geometry("300x150")
        self.progress_window.transient(self.root)
        self.progress_window.grab_set()
        
        # Center the progress window
        self.progress_window.geometry("+%d+%d" % (
            self.root.winfo_x() + self.root.winfo_width()//2 - 150,
            self.root.winfo_y() + self.root.winfo_height()//2 - 75
        ))
        
        # Add progress label
        self.progress_label = Label(self.progress_window, text="Initializing...", pady=10)
        self.progress_label.pack()
        
        # Add progress bar
        self.progress_bar = ttk.Progressbar(
            self.progress_window, 
            orient="horizontal",
            length=200, 
            mode="determinate"
        )
        self.progress_bar.pack(pady=10)

    def update_progress(self, value, text):
        """Update progress bar and label"""
        if self.progress_bar and self.progress_label:
            self.progress_bar["value"] = value
            self.progress_label["text"] = text
            self.progress_window.update()

if __name__ == "__main__":
    app = SignLanguageDetector()
    app.run() 