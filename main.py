from age_weight_regression import ImageToTwoParamsModel
from gender_classifier import BinaryClassifier
import torch
import os
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import sys

gender_classifier = BinaryClassifier().to('cuda')
gender_classifier.load_state_dict(torch.load('gender_classifier.pth'))
age_weight_predictor = ImageToTwoParamsModel().to('cuda')
age_weight_predictor.load_state_dict(torch.load('age_weight_predictor.pth'))

def showImage(image):
    plt.figure(figsize=(30, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(image[image.shape[0]//2,:,:], cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(image[:, image.shape[1]//2, :], cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(image[:, :, image.shape[2]//4], cmap='gray')
    # plt.show()
    plt.savefig('generated_image.png')
    
def calculate_age_weight(image):
    with torch.no_grad():
        age_weight_predictor.eval()
        data = image.unsqueeze(dim=1).to(torch.float32)
        age_weight_sum = torch.zeros(2).to('cuda')
        count = 0

        for batch in range(0, len(data), 8):
            batch_data = data[batch:batch + 8].to('cuda')
            age_weight_pred = age_weight_predictor(batch_data)
            age_weight_sum += age_weight_pred.sum(dim=0)
            count += batch_data.size(0)

    age_weight_avg = age_weight_sum / count
    return age_weight_avg.cpu()

def calculate_gender(image):
    with torch.no_grad():
        gender_classifier.eval()
        data = image.unsqueeze(dim=1).to(torch.float32)
        gender_sum = torch.zeros(1).to('cuda')
        count = 0
        for batch in range(0, len(data), 8):
            batch_data=data[batch:batch + 8].to('cuda')
            gender_pred=gender_classifier(batch_data)
            gender_sum+=gender_pred.sum(dim=0)
            count+=batch_data.size(0)
    
    gender_avg=gender_sum/count
    return gender_avg.cpu()

def generate(dir, target_age, target_weight, target_gender):
    data = torch.load(dir).unsqueeze(dim=1).to(torch.float32)
    ret = data.clone()
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    progress_bar = tqdm(range(0, len(data), 8))

    for batch in progress_bar:
        batch_data = ret[batch:batch + 8].to('cuda').requires_grad_()
        image_optimizer = torch.optim.Adam([batch_data], lr=0.0002, betas=(0.5, 0.999))
        epochs = 200

        for epoch in range(epochs):
            age_weight_predictor.eval()
            gender_classifier.eval()
            age_weight_pred = age_weight_predictor(batch_data)
            gender_pred = gender_classifier(batch_data)

            target_ages_weights = torch.ones_like(age_weight_pred)
            target_ages_weights[:, 0] *= target_age
            target_ages_weights[:, 1] *= target_weight

            target_genders = torch.ones_like(gender_pred) * target_gender

            # print(gender_pred.min().item(), gender_pred.max().item(), target_genders.min().item(), target_genders.max().item(), batch_data.min().item(), batch_data.max().item(), data[batch:batch + 8].min().item(), data[batch:batch + 8].max().item())

            loss = mse_loss(age_weight_pred, target_ages_weights.to('cuda')) + bce_loss(gender_pred, target_genders.to('cuda')) + bce_loss(batch_data, data[batch:batch + 8].to('cuda'))
            loss.backward()
            image_optimizer.step()
            batch_data.data.clamp_(0, 1)

            progress_bar.set_description(f"Epoch: {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Batch: {batch + 1}/{len(data)}")
        ret[batch:batch + 8] = batch_data.detach().cpu()
    return ret.cpu().squeeze()

if __name__ == "__main__":
    file_path=sys.argv[1]
    generated_image= generate(file_path, 80, 60, 1)
    torch.save(generated_image, 'generated_'+os.path.basename(file_path))
    # generated_image=torch.load("./generated_3d_image_070Y_83_F.pth")
    showImage(generated_image)
    print("Age and weight prediction: ", calculate_age_weight(generated_image))
    print('Gender prediction: ',calculate_gender(generated_image))
