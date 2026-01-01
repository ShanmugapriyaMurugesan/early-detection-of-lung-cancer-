import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Path to your saved model
MODEL_PATH = 'dataset_improved.h5'
IMAGE_SIZE = (350, 350)

# Class indices dictionary based on your training data folders
CLASS_INDICES = {
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib': 0,
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa': 1,
    'normal': 2,
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa': 3
}

# Map detailed classes to simple diagnosis
SIMPLE_LABELS = {
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib': 'Affected',
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa': 'Affected',
    'normal': 'Not Affected',
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa': 'Affected'
}

class LungCancerDetectorApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Lung Cancer Detection")
        self.root.geometry("700x800")
        self.root.resizable(False, False)

        # Load background image
        self.bg_image = Image.open("dreamstime_l_137230671.jpg")  # Replace with your background image path
        self.bg_image = self.bg_image.resize((700, 800), Image.Resampling.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)

        # Create canvas for background
        self.canvas = tk.Canvas(root, width=500, height=600)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")

        # Load model
        self.model = load_model("/dataset_improved.h5")

        # Image label (on canvas)
        self.img_label = tk.Label(root,bg=None, borderwidth=0, highlightthickness=0)
        self.canvas.create_window(350, 250, window=self.img_label,anchor="center")

        # Result label (on canvas)
        self.result_label = tk.Label(root, text="", font=("Arial", 16), fg="blue", bg=self.canvas["bg"])
        self.canvas.create_window( 350, 500, window=self.result_label,anchor="center")

        # Browse button (on canvas)
        browse_button = tk.Button(root, text="Browse Image", command=self.browse_image, width=40,
                                  bg="#4CAF50", fg="white", font=("Arial", 12), relief="raised")
        self.canvas.create_window(350, 550,  window=browse_button,anchor="center")

        # Clear button (on canvas)
        clear_button = tk.Button(root, text="Clear", command=self.clear_image, width=40,
                                 bg="#f44336", fg="white", font=("Arial", 12), relief="raised")
        self.canvas.create_window(350, 600, window=clear_button,anchor="center")

        self.img_path = None

    def browse_image(self):
        file_types = [("Image files", "*.jpg *.jpeg *.png")]
        self.img_path = filedialog.askopenfilename(title="Select an image", filetypes=file_types)
        if not self.img_path:
            return

        # Display the selected image resized to fit the label
        self.display_image(self.img_path)

        # Predict and display result
        self.predict_image()

    def display_image(self, img_path):
        img = Image.open(img_path)
        img.thumbnail((400, 400))  # Resize keeping aspect ratio
        self.img_tk = ImageTk.PhotoImage(img)
        self.img_label.configure(image=self.img_tk)
        self.result_label.config(text="")  # Clear previous result

    def predict_image(self):
        try:
            # Preprocess image
            img = image.load_img(self.img_path, target_size=IMAGE_SIZE)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            labels = list(CLASS_INDICES.keys())
            predicted_label = labels[predicted_class]

            # Get simple diagnosis label
            simple_label = SIMPLE_LABELS.get(predicted_label, "Unknown")

            self.result_label.config(text=f"Diagnosis: {simple_label}")

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

    def clear_image(self):
        self.img_label.configure(image="")
        self.result_label.config(text="")
        self.img_path = None


if __name__ == "__main__":
    root = tk.Tk()
    app = LungCancerDetectorApp(root)
    root.mainloop()
