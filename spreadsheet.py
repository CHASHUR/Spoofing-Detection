import cv2
import numpy as np
import os
import zipfile
import pandas as pd
from matplotlib import pyplot as plt

def extract_prnu_snu(image):
    prnu = cv2.fastNlMeansDenoising(image, None, h=7, templateWindowSize=21, searchWindowSize=21)
    snu = image.astype(np.float32) - prnu.astype(np.float32)
    return prnu, snu

def compute_noise_level(image, mask=None):
    if mask is not None:
        noise_level = np.std(image[mask > 0])
    else:
        noise_level = np.std(image)
    return noise_level

def otsu_threshold(image):
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def noise_analysis(foreground_noise, background_noise):
    noise_difference = abs(foreground_noise - background_noise)
    return noise_difference  # Adjust threshold as needed

def detect_moire_pattern(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # Avoid log(0) issues

    moire_pattern = np.sum(magnitude_spectrum > np.mean(magnitude_spectrum) + 3 * np.std(magnitude_spectrum))
    
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    return moire_pattern

def process_image(image):
    prnu, snu = extract_prnu_snu(image)
    foreground_mask = otsu_threshold(image)
    background_mask = otsu_threshold(prnu)
    foreground_noise = compute_noise_level(image, foreground_mask)
    background_noise = compute_noise_level(prnu, background_mask)
    noise_difference = noise_analysis(foreground_noise, background_noise)
    moire_pattern = detect_moire_pattern(image)
    
    return foreground_noise, background_noise, noise_difference, moire_pattern

def detect_fake_images(zip_file_path):
    results_list = []
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        image_files = [name for name in zip_ref.namelist() if name.endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in image_files:
            with zip_ref.open(image_file) as file:
                image_bytes = np.frombuffer(file.read(), np.uint8)
                image = cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)

            foreground_noise, background_noise, noise_difference, moire_pattern = process_image(image)

            if noise_difference < 0.23 and moire_pattern < 100:  # Adjust thresholds as needed
                result = "Real"
            else:
                result = "Fake"

            results_list.append({"Image": image_file,
                                 "Foreground_Noise": foreground_noise,
                                 "Background_Noise": background_noise,
                                 "Noise_Difference": noise_difference,
                                 "Moiré_Pattern": moire_pattern,
                                 "Result": result})

    results_df = pd.DataFrame(results_list)
    return results_df

def main():
    zip_file_path = r'C:\Users\carin\Downloads\SEM 7\minor project\real.zip'
    results_df = detect_fake_images(zip_file_path)
    output_file_path = os.path.join(os.getcwd(), 'image_analysis_results_with_moire.xlsx')
    results_df.to_excel(output_file_path, index=False)

if __name__ == "__main__":
    main()
