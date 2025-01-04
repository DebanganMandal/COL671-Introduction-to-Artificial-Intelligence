import sys

#Any additional sklearn import will flagged as error in autograder
from sklearn.metrics import f1_score
import sklearn.metrics as metrics
import skimage.metrics as image_metrics  #SSIM

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid
from collections import defaultdict

from scipy.stats import norm
import csv

import pickle


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim1=512, hidden_dim2=256, latent_dim=2):
        super(VAE, self).__init__()
        
        # encoder part
        self.cnn1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.cnn2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc3 = nn.Linear(32*7*7, hidden_dim1)
        self.fc4 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc51 = nn.Linear(hidden_dim2, latent_dim)
        self.fc52 = nn.Linear(hidden_dim2, latent_dim)

        # decoder part
        self.fc6 = nn.Linear(latent_dim, hidden_dim2)
        self.fc7 = nn.Linear(hidden_dim2, hidden_dim1)
        self.fc8 = nn.Linear(hidden_dim1, 32*7*7)
        self.tcnn9 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.tcnn10 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, output_padding=1, padding=1)
        
    def encoder(self, x):
        h = F.relu(self.cnn1(x))
        h = F.relu(self.cnn2(h))
        h = F.relu(self.fc3(h.view(-1, 32*7*7)))
        h = F.relu(self.fc4(h))
        return self.fc51(h), self.fc52(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return eps * std + mu 
        
    def decoder(self, z): 
        h = F.relu(self.fc6(z))
        h = F.relu(self.fc7(h))
        h = F.relu(self.fc8(h)).view(-1, 32, 7, 7)
        h = F.relu(self.tcnn9(h))
        return torch.sigmoid(self.tcnn10(h))
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar


class GMM:
    def __init__(self, K=3, n_iter=100, tol=1e-6, init_means=None):
        self.K = K
        self.n_iter = n_iter
        self.tol = tol
        self.init_means = init_means
        self.labels = {}
        self.labels_list = []

    def initialize_parameters(self, X):
        N, D = X.shape
        if self.init_means is not None:
            self.means = np.array(list(self.init_means.values()))
            self.labels_list = np.array(list(self.init_means.keys()))
            self.labels = {index: value for index, value in enumerate(self.labels_list)}
        else:
            self.means = X[np.random.choice(N, self.K, replace=False)]
        self.covariances = [np.eye(D) for _ in range(self.K)]
        self.weights = np.ones(self.K) / self.K

    def gaussian_pdf(self, X, mean, cov):
        D = X.shape[1]
        cov_det = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)
        norm_const = 1.0 / (np.power(2 * np.pi, D / 2) * np.sqrt(cov_det))
        X_mean = X - mean
        result = norm_const * np.exp(-0.5 * np.sum(X_mean @ cov_inv * X_mean, axis=1))
        return result

    def E_step(self, X):
        N, _ = X.shape
        responsibility = np.zeros((N, self.K))
        for k in range(self.K):            
            pdf = self.gaussian_pdf(X, self.means[k], self.covariances[k])
            responsibility[:, k] = self.weights[k]*pdf
        R = responsibility.sum(axis=1, keepdims=True)
        responsibility = responsibility/R
        return responsibility

    def M_step(self,X,responsilibity):
        N,D = X.shape
        for k in range(self.K):
            resp_k = responsilibity[:, k]
            Nk = resp_k.sum()
            self.weights[k] = Nk/N # Update weights
            self.means[k] = (X * resp_k[:, np.newaxis]).sum(axis=0)/Nk #Update means
            X_mean = X-self.means[k]
            self.covariances[k] = (resp_k[:, np.newaxis, np.newaxis] * (X_mean[:, :, np.newaxis] * X_mean[:, np.newaxis, :])).sum(axis=0)
            self.covariances[k] = self.covariances[k]/Nk

    def fit(self, X):
        self.initialize_parameters(X)
        log_likelihood_old = None

        for i in range(self.n_iter):
            resposibility = self.E_step(X)
            self.M_step(X, resposibility)
            log_likelihood = np.sum(np.log(np.sum(resposibility, axis=1)))
            if log_likelihood_old is not None and abs(log_likelihood - log_likelihood_old)<self.tol:
                break
            log_likelihood_old = log_likelihood
        return self

    def predict(self, X):
        responsibility = self.E_step(X)
        max_indices = np.argmax(responsibility, axis=1)
        predictions = [self.labels[idx] for idx in max_indices]
        return np.array(predictions)


def loss_function(x, recon_x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def show_reconstruction(model, val_loader, n=10):
    model.eval()
    data, labels = next(iter(val_loader))
    
    data = data.to(device)
    recon_data, _, _ = model(data)
    
    fig, axes = plt.subplots(2, n, figsize=(15, 4))
    for i in range(n):
        # Original images
        axes[0, i].imshow(data[i].cpu().numpy().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        # Reconstructed images
        axes[1, i].imshow(recon_data[i].cpu().view(28, 28).detach().numpy(), cmap='gray')
        axes[1, i].axis('off')
    plt.show()

def plot_2d_manifold(model, n=20, latent_dim=2, digit_size=28, device='cuda'):
    figure = np.zeros((digit_size * n, digit_size * n))

    # Generate a grid of values between 0.05 and 0.95 percentiles of a normal distribution
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    model.eval()  # Set VAE to evaluation mode
    with torch.no_grad():
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                # std = torch.exp(logvar)
                # eps = torch.randn_like(std) 
                # z_sample = mu + eps * std 
                z_sample = torch.tensor([[xi, yi]], device=device).float()  # 2D latent space
                
                digit = model.decoder(z_sample).cpu().numpy().squeeze()

                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gnuplot2')
    plt.axis('off')
    plt.show()

def extract_latent_means(model, train_loader, device='cuda'):
    model.eval()
    latents = []
    labels = []
    
    with torch.no_grad():
        for x, y in train_loader:
                        
            x = x.to(device)
            mu, logvar = model.encoder(x)
            z = model.reparameterize(mu, logvar)
            latents.append(z.cpu().numpy()) 
            labels.append(y.cpu().numpy())  
    
    latents = np.concatenate(latents, axis=0)  
    labels = np.concatenate(labels, axis=0)  
    return latents, labels

def reconstruct(data_loader, model, device, save_file = "vae_reconstructed.npz"):
    model.eval()
    model.to(device)
    array = []
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels= imgs.to(device), labels.to(device)
            reconstructed_x,_, _= model(imgs)
            array.append(reconstructed_x.cpu().numpy().squeeze())
    np.savez(save_file, data = array)


if __name__ == "__main__": 
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    arg3 = sys.argv[3] if len(sys.argv) > 3 else None
    arg4 = sys.argv[4] if len(sys.argv) > 4 else None
    arg5 = sys.argv[5] if len(sys.argv) > 5 else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(sys.argv)==4:### Running code for vae reconstruction.
        path_to_test_dataset_recon = arg1
        test_reconstruction = arg2
        vaePath = arg3

        test_data = np.load(path_to_test_dataset_recon)

        test_images = test_data['data'].astype(np.float32)/255.0
        test_labels = test_data['labels']

        test_images = np.expand_dims(test_images, axis=-1)

        test_images = torch.tensor(test_images).permute(0,3,1,2)
        test_labels = torch.tensor(test_labels)

        test_dataset = TensorDataset(test_images, test_labels)
        
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        vae = VAE()
        vae.load_state_dict(torch.load(vaePath))
        vae.to(device)        

        reconstruct(test_loader, vae, device)

        
    elif len(sys.argv)==5:###Running code for class prediction during testing
        path_to_test_dataset = arg1
        test_classifier = arg2
        vaePath = arg3
        gmmPath = arg4

        test_data = np.load(path_to_test_dataset)

        test_images = test_data['data'].astype(np.float32)/255.0
        test_labels = test_data['labels']

        test_images = np.expand_dims(test_images, axis=-1)

        test_images = torch.tensor(test_images).permute(0,3,1,2)
        test_labels = torch.tensor(test_labels)

        test_dataset = TensorDataset(test_images, test_labels)
        
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        model = VAE().to(device)
        model.load_state_dict(torch.load(vaePath))
        model.eval()

        gmm = GMM()
        data = None
        with open(gmmPath, mode='rb') as f:
            data = pickle.load(f)

        if data is not None:
            gmm = data

        X_test, labels_test = extract_latent_means(model, test_loader)

        predictions = gmm.predict(X_test)
        print(predictions)

        output_csv = "vae.csv"
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Predicted_Label'])
            for label in predictions:
                writer.writerow([label])


    else:### Running code for training. save the model in the same directory with name "vae.pth"
        path_to_train_dataset = arg1
        path_to_val_dataset = arg2
        trainStatus = arg3
        vaePath = arg4
        gmmPath = arg5

        train_data = np.load(path_to_train_dataset)
        val_data = np.load(path_to_val_dataset)

        train_images = train_data['data'].astype(np.float32)/255.0
        train_labels = train_data['labels']

        val_images = val_data['data'].astype(np.float32)/255.0
        val_labels = val_data['labels']

        train_images = np.expand_dims(train_images, axis=-1)
        val_images = np.expand_dims(val_images, axis=-1)

        train_images = torch.tensor(train_images).permute(0,3,1,2)
        val_images = torch.tensor(val_images).permute(0,3,1,2)
        train_labels = torch.tensor(train_labels)
        val_labels = torch.tensor(val_labels)

        num_epochs = 300

        train_dataset = TensorDataset(train_images, train_labels)
        val_dataset = TensorDataset(val_images, val_labels)

        train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=True)

        model = VAE().to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)

        model.train()
        for epoch in range(num_epochs):
            overall_loss = 0
            for x,_ in train_loader:
                x = x.to(device)

                optimizer.zero_grad()

                recon_x, mu, logvar = model(x)
                loss = loss_function(x, recon_x, mu, logvar)

                loss.backward()
                optimizer.step()
                
                overall_loss += loss.item()

            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {overall_loss/len(train_loader):.4f}')

        torch.save(model.state_dict(),vaePath)

        model.eval()
        X_train, labels_train = extract_latent_means(model, train_loader)
        X_val, labels_val = extract_latent_means(model, val_loader)

        class_latent_means = defaultdict(list)

        # Collect latent means for each class
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)  
                mu, _ = model.encoder(images)  
                for i, label in enumerate(labels):
                    class_latent_means[label.item()].append(mu[i].cpu())  # Store latent mean by class

        class_means = {}
        for label, latents in class_latent_means.items():
            latents_tensor = torch.stack(latents)  
            class_means[label] = latents_tensor.mean(dim=0)  
        class_means_np = {label: mean_tensor.cpu().numpy() for label, mean_tensor in class_means.items()}

        gmm = GMM(init_means=class_means_np)
        gmm.fit(X_train)
        predictions = gmm.predict(X_val)

        with open(gmmPath, 'wb') as f:
            pickle.dump(gmm, f)



    print(f"arg1:{arg1}, arg2:{arg2}, arg3:{arg3}, arg4:{arg4}, arg5:{arg5}")



