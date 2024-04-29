import torch

def calculate_mean_std(loader):
    # Variables to accumulate the mean and std values
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images_count = 0

    for images, _ in loader:
        # Update total count of images
        batch_samples = images.size(0)
        total_images_count += batch_samples

        # Reshape the images to (batch_size, channels, -1)
        # -1 flattens the remaining two dimensions (width and height)
        images = images.view(batch_samples, images.size(1), -1)
        
        # Sum up the mean values for each channel
        mean += images.mean(2).sum(0)
        # Sum up the std values for each channel
        std += images.std(2).sum(0)

    # Average the mean and std by the number of images
    mean /= total_images_count
    std /= total_images_count

    return mean, std