import os
from tkinter import Tk, Label, Button

import pygame
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np

from DeepLearning.nn import Net, transform

WHITE = (255,255,255)
BLACK = (0,0,0)

class GUI:
    def __init__(self, model_path=None, image_size=(96,96)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = Net(10).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        else:
            print("Aucun modèle trouvé : mode collecte d'images uniquement")
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.image_size= image_size
        self.screen = pygame.display.get_surface()

        self.drawing = False
        self.last_pos = None

    #Labellisation et stocker les images d'entraînement
    def save_label(self, label, path='DeepLearning/dataset/'):
        name = f'{label}_' + str(len(os.listdir(f'{path}{label}'))+1)
        print(f"Saving {name}.png")
        image = pygame.surfarray.array3d(self.screen)
        image = np.flipud(image) #Invert along Y axis cause pygame invert the y axis when taking the screen...
        image = np.rot90(image, k=-1).copy() #rotation poura voir l'image à l'endroit comme on veut

        image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() #on veut avoir en première dimension le nombre de canaux
        #Save the image but don't apply self.transform
        save_image(image_tensor, f'{path}{label}/{name}.png')

    def display_prediction(self, prediction):
        popup = Tk()
        popup.wm_title("Prediction")

        label = Label(popup, text=f"Predicted Digit : {prediction}", font=("Helvetica", 16))
        label.pack(side="top", fill="x", pady=20, padx=20)

        button = Button(popup, text="OK", command=popup.destroy)
        button.pack(side="bottom", pady=10)

        popup.mainloop()

    def run(self):
        self.screen.fill(BLACK)
        while True:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.drawing = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.drawing = False
                    self.last_pos = None
                elif event.type == pygame.MOUSEMOTION:

                    if self.drawing:
                        pos = pygame.mouse.get_pos()
                        if self.last_pos:
                            pygame.draw.line(self.screen, WHITE, self.last_pos, pos, 10)
                        self.last_pos = pos 

                elif event.type == pygame.K_SPACE:
                    self.screen.fill(BLACK)

                elif event.type == pygame.KEYDOWN:
                    #press enter
                    if event.key == pygame.K_RETURN:

                        #get label
                        self.save_label(4)

                        if self.model is not None:

                            image = pygame.surfarray.array3d(self.screen)
                            image = np.flipud(image) #Invert along Y axis
                            image = np.rot90(image, k=-1).copy()

                            image = self.transform(image).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                output = self.model(image)
                                #apply softmax
                                output = torch.softmax(output, dim=1)
                                print(output)
                                _, prediction = output.max(1) #pas besoin de la valeur du score max donc on l'enregistre pas dans une variable

                                print(f"Predicted Digit : {prediction.item()}")
                                self.display_prediction(prediction.item())

                        #clear the screen
                        self.screen.fill(BLACK)
            pygame.display.update()
