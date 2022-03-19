import matplotlib.pyplot as plt


def show_image(img):
    plt.figure(dpi=150)
    plt.imshow(img)
    plt.show()
    plt.close()