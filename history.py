import os
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox


class ImageSwitcher:
    def __init__(self, parent, image_folder='./user_results/history'):
        self.image = None
        self.photo = None
        self.root = tk.Toplevel()
        self.parent = parent
        self.root.title("History")
        self.root.geometry("800x500+320+150")
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if
                            f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        self.image_index = 0
        if not self.image_files:
            messagebox.showwarning("Warning", "No image files found in the specified folder.")
            self.root.destroy()
            self.parent.deiconify()
            return

        # Create a label to display the image
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        # 创建底部框架
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side='bottom', pady=0, fill='y', expand=True)

        # Create a button to switch to the next image
        self.next_button = tk.Button(bottom_frame, text="Next", command=self.next_image,width=20)
        self.next_button.pack(side='left', padx=20)

        self.close_button = tk.Button(bottom_frame, text="Close", command=self.close,width=20)
        self.close_button.pack(side='right',padx=20)
        # Load the first image
        self.load_image()

    def load_image(self):
        image_path = os.path.join(self.image_folder, self.image_files[self.image_index])
        self.image = Image.open(image_path)
        self.image = self.image.resize((800, 400))
        self.photo = ImageTk.PhotoImage(self.image)
        self.image_label.config(image=self.photo)

    def next_image(self):
        self.image_index = (self.image_index + 1) % len(self.image_files)
        self.load_image()

    def show(self):
        self.root.mainloop()

    def close(self):
        self.root.destroy()
        self.parent.deiconify()
