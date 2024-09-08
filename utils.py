# Function to resize images to the same height
def resize_image(image, target_height):
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(target_height * aspect_ratio)
    resized_image = image.resize((new_width, target_height))
    return resized_image