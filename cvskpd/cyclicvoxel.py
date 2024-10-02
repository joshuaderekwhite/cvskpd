import numpy as np
import cv2

def resize_image(image, new_height, new_width, kernel_size=1):
    # Mask of non-zeros
    mask = image!=0 # Use a >tolerance for a tolerance defining black border

    # Mask of non-zero rows and columns
    mask_row = mask.any(1)
    mask_col = mask.any(0)

    # First, last indices among the non-zero rows
    sr0,sr1 = mask_row.argmax(), len(mask_row) - mask_row[::-1].argmax()

    # First, last indices among the non-zero columns
    sc0,sc1 = mask_col.argmax(), len(mask_col) - mask_col[::-1].argmax()

    # Finally slice along the rows & cols with the start and stop indices to get 
    # cropped image. Slicing helps for an efficient operation.
    image = image[sr0:sr1, sc0:sc1]
    # image_area = {"height": sr1-sr0, "width": sc1-sc0}

    # Original dimensions
    original_height, original_width = image.shape[:2]

    # Create an empty array for the resized image
    resized_image = np.zeros((new_height, new_width), dtype=np.float64)

    # Calculate the ratio of old dimensions to new dimensions
    row_ratio = original_height / new_height
    col_ratio = original_width / new_width

    # Piecewise Interpolation
    for i in range(new_height):
        for j in range(new_width):
            # Find the nearest pixel in the original image
            original_i = int(i * row_ratio)
            original_j = int(j * col_ratio)

            # Assign the pixel value
            resized_image[i, j] = image[original_i, original_j]

    # Filter the image to account for movement
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
    resized_image = cv2.filter2D(resized_image, -1, kernel)

    return resized_image
# remove image_area , image_area

def resize_image_3d(image, new_height, new_width, new_depth, kernel_size=1):
    # Original dimensions
    original_height, original_width, original_depth = image.shape

    # Create an empty array for the resized image with the new dimensions
    resized_image = np.zeros((new_height, new_width, new_depth), dtype=np.float64)

    # Calculate the ratio of old dimensions to new dimensions
    row_ratio = original_height / new_height
    col_ratio = original_width / new_width
    depth_ratio = original_depth / new_depth

    # Piecewise Interpolation
    for i in range(new_height):
        for j in range(new_width):
            for k in range(new_depth):
                # Find the nearest pixel in the original image for each dimension
                original_i = int(i * row_ratio)
                original_j = int(j * col_ratio)
                original_k = int(k * depth_ratio)

                # Assign the pixel value from the original image
                resized_image[i, j, k] = image[original_i, original_j, original_k]

    # Filter the image to account for movement
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
    for k in range(new_depth):
        resized_image[:, :, k] = cv2.filter2D(resized_image[:, :, k], -1, kernel)

    return resized_image

def cyclic_tensor(tensor, new_dims, kernel_size=1, bootstrap_method=None):
    # Currently only works for 2D and 3D tensors and bootstrapping views for 3D tensors
    if bootstrap_method is None:
        match len(new_dims):
            case 3:
                tensor_out = resize_image_3d(tensor, new_dims[0], new_dims[1], new_dims[2], kernel_size=kernel_size)
            case 2:
                tensor_out = resize_image(tensor, new_dims[0], new_dims[1], kernel_size=kernel_size)
    else:
        # bootstrap_method = {
        #   "keys": ~Names of each of the bootstrap views~, 
        #   "method": ~Method of bootstrapping~,
        #   "kwargs": ~Keyword arguments for the bootstrapping method~
        # }
        tensor_out = {}
        for i, b in enumerate(bootstrap_method["keys"]):
            slices = [slice(None)] * len(tensor.shape)
            vals = bootstrap_method["method"](tensor.shape[i], **bootstrap_method["kwargs"])
            slices[i] = vals
            tensor_out[b] = resize_image(np.average(tensor[tuple(slices)], axis=i), new_dims[0], new_dims[1], kernel_size=kernel_size)
    return tensor_out

