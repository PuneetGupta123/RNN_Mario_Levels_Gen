from PIL import Image
import os

# Set the root directory
outdir = "/home/akash/Desktop/Major_2/Major_image"
newImage = "newImage.jpeg"
def long_slice(baseDir, outdirr, sliceHeight, sliceWidth):
    Image.open(baseDir+"/slice_2992_224_7.jpg").convert('RGB').save(newImage)
    img = Image.open(baseDir+"/"+newImage) # Load image
    imageWidth, imageHeight = img.size # Get image dimensions
    left = 0 # Set the left-most edge
    upper = 200 # Set the top-most edge
    while (upper > 0):
        while (left < imageWidth):
            # If the bottom and right of the cropping box overruns the image.
            if (upper + sliceHeight > imageHeight and \
                left + sliceWidth > imageWidth):
                bbox = (left, upper, imageWidth, imageHeight)
            # If the right of the cropping box overruns the image
            elif (left + sliceWidth > imageWidth):
                bbox = (left, upper, imageWidth, upper + sliceHeight)
            # If the bottom of the cropping box overruns the image
            elif (upper + sliceHeight > imageHeight):
                bbox = (left, upper, left + sliceWidth, imageHeight)
            # If the entire cropping box is inside the image,
            # proceed normally.
            else:
                bbox = (left, upper, left + sliceWidth, upper + sliceHeight)
            working_slice = img.crop(bbox) # Crop image based on created bounds
            # Save your new cropped image.
            working_slice.save(os.path.join(outdirr, 'slice_' + str(upper) + '_' + str(left) + '.jpg'))
            left += sliceWidth + 1
        upper -= (sliceHeight + 1)
        left = 0

if __name__ == '__main__':
    long_slice("/home/akash/Desktop/Major_2", outdir, 15, 15)
