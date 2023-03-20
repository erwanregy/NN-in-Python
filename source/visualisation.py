from datatypes import Matrix, Image

ascii_scale = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$$"
pixel_width = 2

def print_image(image: Image) -> None:
    if isinstance(image[0][0], int):
        image: Matrix = [[value / 255.0 for value in row] for row in image]
    
    print("+" + "-" * len(image[0]) * pixel_width + "+")
    for row in image:
        print("|", end="")
        for value in row:
            value = max(0.0, min(1.0, value))
            value = int(value * (len(ascii_scale) - 1))
            print(ascii_scale[value] * pixel_width, end="")
        print("|")
    print("+" + "-" * len(image[0]) * pixel_width + "+")
    

if __name__ == "__main__":
    from math import sqrt
    
    width = int(sqrt(256))
    image = [[i + j * width for i in range(width)] for j in range(width)]
    
    matrix = [[value / 255 for value in row] for row in image]
    
    print_image(image)
    print_image(matrix)

    import matplotlib.pyplot as plt
    
    plt.imshow(image, cmap="gray") 
    plt.show()
    
    plt.imshow(matrix, cmap="gray")
    plt.show()
    
    input("Press enter to exit...")
    