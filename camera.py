import cv2
import numpy as np
import os
import zipfile
import pandas as pd

def extract_prnu_snu(image):
    # Estimate PRNU pattern using Non-Local Means Denoising
    prnu = cv2.fastNlMeansDenoising(image, None, h=7, templateWindowSize=21, searchWindowSize=21)

    # Compute SNU pattern as the difference between original and
    # denoised image
    snu = image - prnu

    return prnu, snu

def compute_noise_level(image, mask=None, downsample_factor=4):
    # Downsample the image and mask
    if downsample_factor > 1:
        image = cv2.resize(image, (image.shape[1] // downsample_factor, image.shape[0] // downsample_factor))
        if mask is not None:
            mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]))

    # Compute noise level of the image within the specified mask region
    if mask is not None:
        noise_level = np.std(image[mask.astype(bool)])
    else:
        noise_level = np.std(image)
    return noise_level

def otsu_threshold(image):
    # Apply Otsu's thresholding to the image
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def noise_analysis(foreground_noise, background_noise):
    # Implement noise analysis logic here
    # Compare noise levels in foreground and background regions
    # Determine if the difference is significant enough to indicate manipulation
    noise_difference = abs(foreground_noise - background_noise)
    return noise_difference  # Adjust threshold as needed

def process_image(image):
    # Extract PRNU and SNU patterns
    prnu, snu = extract_prnu_snu(image)

    # Compute Otsu's threshold masks for foreground and background
    foreground_mask = otsu_threshold(image)
    background_mask = otsu_threshold(prnu)

    # Compute noise level in foreground and background regions
    foreground_noise = compute_noise_level(image, foreground_mask, downsample_factor=4)
    background_noise = compute_noise_level(prnu, background_mask, downsample_factor=4)

    # Perform pattern analysis and noise analysis
    noise_difference = noise_analysis(foreground_noise, background_noise)

    return foreground_noise, background_noise, noise_difference

def detect_fake_images(zip_file_path):
    # Initialize an empty list to store results
    results_list = []

    # Extract images from the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        image_files = [name for name in zip_ref.namelist() if name.endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in image_files:
            with zip_ref.open(image_file) as file:
                image_bytes = np.frombuffer(file.read(), np.uint8)
                image = cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)

            # Process the image
            foreground_noise, background_noise, noise_difference = process_image(image)

            # Determine the result (Real/Fake)
            if noise_difference < 0.23:
                result = "Real"
            else:
                result = "Fake"

            # Append results to the list
            results_list.append({"Image": image_file,
                                 "Foreground_Noise": foreground_noise,
                                 "Background_Noise": background_noise,
                                 "Noise_Difference": noise_difference,
                                 "Result": result})

    # Convert the list to a DataFrame
    results_df = pd.DataFrame(results_list)

    return results_df

def main():
    # Replace 'zip_file_path' with the actual path to your zipped file
    zip_file_path = r'C:\Users\carin\Downloads\SEM 7\COLLEGE\Extras\minor project\fake.zip'

    # Detect if the images are fake or real
    results_df = detect_fake_images(zip_file_path)

    # Save results to a spreadsheet in the current working directory
    output_file_path = os.path.join(os.getcwd(), 'image_analysis_results3.xlsx')
    results_df.to_excel(output_file_path, index=False)

if __name__ == "__main__":
    main()
