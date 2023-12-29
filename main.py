# imports:
import torch
import numpy as np
from torchvision.utils import save_image
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.functional import normalize
import json
from PIL import Image
import glob
# Importing the generator:
import GANLatentDiscovery.loading
import GANLatentDiscovery.models.gan_load
import GANLatentDiscovery.models.BigGAN.BigGAN
import GANLatentDiscovery.models.BigGAN.utils

#deformator, generator, shift_predictor = GANLatentDiscovery.loading.load_from_dir('./GANLatentDiscovery/', G_weights='./models/G_ema.pth')
generator = GANLatentDiscovery.loading.load_generator(json.load(open('./GANLatentDiscovery/args.json')),'./models/G_ema.pth')

# Importing the WideResNet-50 robust ImageNet model:             
from robustbench.utils import load_model
model = load_model(model_name='Salman2020Do_R18', dataset='imagenet', threat_model='Linf')


# Create a random input for the Generator
def create_input():
    input_vector = torch.randn((1,128))
    input_vector = input_vector.requires_grad_(False)

    return input_vector


# Defining an Optimizer
def define_optimizer(input):
    optimizer = optim.Adam([input], lr = 0.2) 
    return optimizer                                                            

# Plot the input image:
def visualize(input):  
    npimg = input.detach().numpy().squeeze(0)
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()


# "Training": (Not really. We optimize the input of the Image Generator, not the model itself) 
def training2(input, optimizer, class_number):
    
    for epoch in range(1000): 
        # zero the parameter gradients
        optimizer.zero_grad()
    
        generated_image = generator(input)

        # forward + backward + optimize
        # output = F.softmax(model(generated_image),dim = 1)     #Probabilities
        output = model(generated_image)                         #Logits      
        #loss = 1 - output[0][class_number]     
        loss = - output[0][class_number]      
        loss.backward()

        optimizer.step()

        if (epoch%50 == 0):
            print("\nEpoch #", epoch,":\n")
            #visualize(generated_image)
            print("Loss: ",loss.item())

    print('Finished Training\n\n\n')

    return generated_image


#################################################################################################################
#Code for checking the specific GoldFish label Logit that the classifier outputs for Goldfish images (Just checking something)

def images_for_inference ():
    goldfish_images = []
    directory_path = './imagenet/*.jpg'
    images = glob.glob(directory_path)
    for image in images:
        img = Image.open(image).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
        ])
        input_image = transform(img)
        goldfish_images.append(input_image)
        
    return goldfish_images

def inference_histogram (input_images):
    goldfish_logits = np.empty(0)
    
    for input in input_images:
        output = model(input)
        #print(output)
        print("#############################################")
        goldfish_logit = output[0][1]
        print("goldfish logit: ", goldfish_logit)
        goldfish_logits = np.append(goldfish_logits, goldfish_logit.detach().numpy())
    
    # Plotting the histogram
    plt.hist(goldfish_logits, bins=10, edgecolor='black')
    # Adding labels and title to the plot
    plt.xlabel('Logit value')
    plt.ylabel('Number of images')
    plt.title('Histogram of the robustbench classifier output logits for Goldfish (10 goldfish images)')

    # Displaying the plot
    plt.show()

#goldfish_images = images_for_inference()
#inference_histogram(goldfish_images)
#################################################################################################################

def calculate_fitnesses(generated_images):
    generated_images_array = np.array(generated_images)
    outputs = model(generated_images_array)
    losses = outputs[:,1]  # Assuming the output has shape (batch_size, num_classes)
    return losses.tolist()


def milestone2():
    input_vector = create_input()
    generated_image = training2(input_vector, define_optimizer(input_vector), 1) #Goldfish's index in the Resent's output - we want an image of a goldfish.
    save_image(generated_image, "goldfish.png")


def milestone3(num_of_generations, generation_size, tournament_size):
    input_vectors = []
    for i in range(generation_size):
        input_vectors.append(create_input())

    # Evolution loop:
    for generation in range(num_of_generations):
        print("finished creating generation " + str(generation))
        fitnesses = []
        new_input_vectors = []
        # compute the fitness of the current generation:
        for input_vector in input_vectors:
           generated_image = generator(input_vector)
           output = model(generated_image)
           loss = - output[0][1]     # class 1 = goldfish
           fitnesses.append(loss.item())

        # just to see if we are going in the right direction as generations go by:
        print("Generation " + str(generation) + " best fitness: " + str(max(fitnesses)))

        # Selection: 
        for tournament in range(generation_size):                 # number of tournaments = generation size

            # draw 10 random vectors out of input_vectors:
            selected_vectors = np.random.choice(input_vectors, size = tournament_size, replace=False)
            # Select the one with the lowest corresponding fitness:
            selected_vector = min(selected_vectors, key=lambda vector: fitnesses[input_vectors.index(vector)])
            # Insert it to a new input_vectors array:
            new_input_vectors.append(selected_vector)

        # Update input_vectors for the next generation:
        input_vectors = new_input_vectors

    # out of the final generation of input vectors, select the one with the best fitness and send it to the generator to show the result:
    final_vector = min(input_vectors, key=lambda vector: fitnesses[input_vectors.index(vector)])
    final_generated_image = generator(final_vector)
    save_image(final_generated_image, "goldfish_with_evolution.png")

#main:
milestone3(20, 30, 5)