import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageEnhance
import easyocr
from transformers import pipeline
from Levenshtein import distance as levenshtein_distance

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Initialize the image captioning pipelines
caption_pipeline_vit_gpt2_coco = pipeline('image-to-text', model='ydshieh/vit-gpt2-coco-en', framework='pt')
caption_pipeline_vit_gpt2_image_captioning = pipeline('image-to-text', model='nlpconnect/vit-gpt2-image-captioning', framework='pt')

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        load_image(file_path)
        analyze_button.configure(state=ctk.NORMAL)
    else:
        messagebox.showerror("Error", "Failed to upload image")

def load_image(file_path):
    global img, img_ctk
    img = Image.open(file_path)
    img.thumbnail((400, 400))
    img_ctk = ctk.CTkImage(light_image=img, size=(400, 400))
    
    # Display the new image
    img_label.configure(image=img_ctk, text="")
    img_label.image = img_ctk
    img_label.file_path = file_path

def calculate_accuracy(ocr_result, reference_text):
    ocr_text = " ".join([text for _, text, _ in ocr_result])
    return 1 - (levenshtein_distance(ocr_text, reference_text) / max(len(ocr_text), len(reference_text)))

def analyze_image():
    if img_label.file_path:
        if option.get() == 'Caption':
            if caption_option.get() == 'ViT-GPT2-COCO':
                result = caption_pipeline_vit_gpt2_coco(img_label.file_path)
            else:
                result = caption_pipeline_vit_gpt2_image_captioning(img_label.file_path)
            output = result[0]['generated_text']
            accuracy_score = None  # No accuracy score for caption
        elif option.get() == 'OCR':
            image = Image.open(img_label.file_path)
            enhancer = ImageEnhance.Contrast(image)
            enhanced_image = enhancer.enhance(2.0)
            ocr_result = reader.readtext(img_label.file_path)  # Pass file path directly
            output = " ".join([text for _, text, _ in ocr_result])
            
            # Ask for reference text to calculate accuracy
            reference_text = simpledialog.askstring("Input", "Please enter the reference text for accuracy calculation:")
            if reference_text:
                accuracy_score = calculate_accuracy(ocr_result, reference_text)
        
        result_text.delete(1.0, ctk.END)
        result_text.insert(ctk.END, output)
        if accuracy_score:
            result_text.insert(ctk.END, f"\n\nAccuracy: {accuracy_score * 100:.2f}%")

def clear_all():
    # Clear the image and reset the label
    img_label.configure(image=None, text="Image Preview")
    img_label.image = None
    img_label.file_path = None

    # Clear the result text area
    result_text.delete(1.0, ctk.END)

    # Disable the analyze button
    analyze_button.configure(state=ctk.DISABLED)

    # Remove the image file reference
    global img, img_ctk
    img = None
    img_ctk = None

# Initialize the main window with customtkinter
ctk.set_appearance_mode("dark")  # Choose "light" or "dark"
ctk.set_default_color_theme("blue")  # Choose a color theme
root = ctk.CTk()
root.title("Image Analysis Tool")

# Create a frame for the header (top bar)
header_frame = ctk.CTkFrame(root)
header_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky='ew')

# Title Label in the header
title_label = ctk.CTkLabel(header_frame, text="Image Analysis Tool", font=("Arial", 24, "bold"))
title_label.grid(row=0, column=0, pady=10, sticky='n')

# Create a frame for the main content
main_frame = ctk.CTkFrame(root)
main_frame.grid(row=1, column=0, padx=20, pady=20, sticky='nsew')

root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

# Image display frame
img_frame = ctk.CTkFrame(main_frame)
img_frame.grid(row=0, column=0, pady=10)

# Clear previous image display
img_label = ctk.CTkLabel(img_frame, text="Image Preview", font=("Arial", 16))
img_label.grid(row=0, column=0, padx=10, pady=10)

# Upload button with modern styling
upload_button = ctk.CTkButton(main_frame, text="Upload Image", command=upload_image)
upload_button.grid(row=1, column=0, pady=10)

# Analysis options
option = ctk.StringVar(value='Caption')
caption_radio = ctk.CTkRadioButton(main_frame, text="Caption", variable=option, value='Caption')
caption_radio.grid(row=2, column=0, pady=5)
ocr_radio = ctk.CTkRadioButton(main_frame, text="OCR", variable=option, value='OCR')
ocr_radio.grid(row=3, column=0, pady=5)

# Caption model options
caption_option = ctk.StringVar(value='ViT-GPT2-COCO')
model1_radio = ctk.CTkRadioButton(main_frame, text="ViT-GPT2-COCO", variable=caption_option, value='ViT-GPT2-COCO')
model1_radio.grid(row=4, column=0, pady=5)
model2_radio = ctk.CTkRadioButton(main_frame, text="ViT-GPT2-Image-Captioning", variable=caption_option, value='ViT-GPT2-Image-Captioning')
model2_radio.grid(row=5, column=0, pady=5)

# Analysis button with modern styling
analyze_button = ctk.CTkButton(main_frame, text="Analyze Image", command=analyze_image, state=ctk.DISABLED)
analyze_button.grid(row=6, column=0, pady=10)

# Clear button with modern styling
clear_button = ctk.CTkButton(main_frame, text="Clear", command=clear_all)
clear_button.grid(row=7, column=0, pady=10)

# Result display with customtkinter styling
result_text = ctk.CTkTextbox(main_frame, height=15, width=70, wrap=ctk.WORD)
result_text.grid(row=0, column=1, rowspan=8, padx=20, pady=10, sticky='nsew')

main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=1)

# Run the main loop
root.mainloop()
