import cv2
import numpy as np

def extract_prnu_snu(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Estimate PRNU pattern using Non-Local Means Denoising
    prnu = cv2.fastNlMeansDenoising(image, None, h=7, templateWindowSize=21, searchWindowSize=21)

    # Compute SNU pattern as the difference between original and denoised image
    snu = image - prnu

    return prnu, snu

def main():
    # Replace 'image_path' with the actual path to your image
    image_path = r'C:\Users\carin\Downloads\SEM 7\COLLEGE\Extras\minor project\fake\(5).jpg'

    # Extract PRNU and SNU patterns
    prnu, snu = extract_prnu_snu(image_path)

    # Display PRNU and SNU patterns
    cv2.imshow('PRNU Pattern', prnu)
    cv2.imshow('SNU Pattern', snu)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
