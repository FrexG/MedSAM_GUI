import threading
import cv2
import numpy as np
import tkinter as tk
import customtkinter
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from infer import infer


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MedSAM App")
        self.canvas_width = 720
        self.canvas_height = 720

        # Create canvas
        self.canvas = tk.Canvas(
            root, width=self.canvas_width, height=self.canvas_height
        )
        self.canvas.pack()

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

        # Initialize variables
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.image = None
        self.tk_image = None

        # Create button for picking image
        self.pick_image_button = customtkinter.CTkButton(
            root, text="Pick Image", command=self.pick_image
        )
        self.pick_image_button.pack()

    def pick_image(self):
        # Open a file dialog to choose an image file
        file_path = filedialog.askopenfilename(
            initialdir=".",
            title="Select Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")],
        )

        if file_path:
            # Load the selected image
            self.load_image(file_path)

    def load_image(self, image_path):
        # Open the image file
        image = Image.open(image_path)

        image = image.resize((720, 720))

        self.image = image
        # Resize the image to fit the canvas
        # if image.width > self.canvas_width or image.height > self.canvas_height:
        #    image.thumbnail((self.canvas_width, self.canvas_height), Image.ANTIALIAS)

        # Create Tkinter-compatible image object
        self.tk_image = ImageTk.PhotoImage(image)

        # Clear canvas and display the image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")

    def on_mouse_press(self, event):
        # Store the starting position of the bounding box
        self.start_x = event.x
        self.start_y = event.y

    def on_mouse_drag(self, event):
        # Update the bounding box as the mouse is dragged
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y, outline="red"
        )

    def on_mouse_release(self, event):
        # Print the coordinates of the bounding box
        print("Bounding box coordinates:")
        print(f"Top-left: ({self.start_x}, {self.start_y})")
        print(f"Bottom-right: ({event.x}, {event.y})")

        # to tensor
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Resize((1024, 1024)),
            ]
        )
        tensor_img = transform(self.image)

        bbox_tensor = torch.tensor([self.start_x, self.start_y, event.x, event.y])

        # Start a new thread for inference
        threading.Thread(
            target=self.run_inference, args=(tensor_img, bbox_tensor)
        ).start()

    def run_inference(self, tensor_img, bounding_box):
        pred_mask = infer(tensor_img.unsqueeze(0), bounding_box)

        print(f"white pix count = {(pred_mask == 255).sum()}")

        pred_mask = pred_mask.squeeze(0).cpu()

        print(f"pred shape = {pred_mask.shape}")

        self.update_gui(pred_mask, tensor_img)

    def update_gui(self, pred_mask, tensor_img):
        # Convert the tensor mask to a PIL image
        print(f"{pred_mask.shape}")
        color_mask = np.zeros((256, 256, 3), dtype=np.uint8)
        color_mask[:, :] = (251, 250, 0)

        color_mask = cv2.bitwise_and(
            color_mask,
            color_mask,
            mask=np.uint8(pred_mask.squeeze(0).numpy()),
        )

        image_cv = np.uint8(
            tensor_img.permute(1, 2, 0).cpu().numpy() * 255
        )  # np.array(self.image)

        color_mask = cv2.resize(color_mask, (self.image.width, self.image.height))

        print(f"{color_mask.shape=}")

        image_cv = cv2.addWeighted(color_mask, 0.8, image_cv, 1, 0)

        # Display the overlay image on the canvas
        image_pil = Image.fromarray(
            image_cv
        )  # cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

        image_tk = ImageTk.PhotoImage(image_pil)

        self.canvas.delete("all")
        self.canvas.create_image(
            0, 0, image=image_tk, anchor="nw"
        )  # tk_overlay, anchor="nw")

        # Store a reference to the PhotoImage object to prevent it from being garbage collected
        self.canvas.image = image_tk


if __name__ == "__main__":
    customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
    customtkinter.set_default_color_theme(
        "blue"
    )  # Themes: blue (default), dark-blue, green

    root = customtkinter.CTk()
    # root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
