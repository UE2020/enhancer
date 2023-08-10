from PIL import Image, ImageEnhance, ImageFilter
import random

def increase_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    brightened_image = enhancer.enhance(factor)
    return brightened_image

def decrease_saturation(image, factor):
    enhancer = ImageEnhance.Color(image)
    desaturated_image = enhancer.enhance(factor)
    return desaturated_image

def apply_subtle_blur(image, radius):
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius))
    return blurred_image

def apply_noise(image, noise_factor):
    width, height = image.size
    pixel_data = image.load()

    for y in range(height):
        for x in range(width):
            r, g, b = pixel_data[x, y]
            noise = int(noise_factor * random.randint(-255, 255))
            pixel_data[x, y] = (
                max(0, min(255, r + noise)),
                max(0, min(255, g + noise)),
                max(0, min(255, b + noise))
            )

    return image

def main():
    input_path = '101543604-186894429.jpg'
    output_path = 'output_image.jpg'
    
    original_image = Image.open(input_path)
    
    # Resize to 3024x4032
    target_size = (4032, 3024)
    resized_image = original_image.resize(target_size, Image.ANTIALIAS)
    
    # Randomized brightness and saturation reduction
    brightness_factor = random.uniform(1.2, 1.5)
    saturation_factor = random.uniform(0.4, 0.6)
    
    # Apply brightness and saturation reduction
    brightened_image = increase_brightness(resized_image, brightness_factor)
    desaturated_image = decrease_saturation(brightened_image, saturation_factor)
    
    # Apply subtle blur
    blurred_image = apply_subtle_blur(desaturated_image, 1)
    
    # Apply noise
    noise_factor = random.uniform(0.05, 0.1)
    noisy_image = apply_noise(blurred_image, noise_factor)
    
    noisy_image.save(output_path)
    print("Image processing complete.")

if __name__ == "__main__":
    main()