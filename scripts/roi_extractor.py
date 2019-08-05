import argparse
import cv2 as cv
import glob
import os
import pathlib

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=False, help="path to input image folder", default='img')
ap.add_argument("-d", "--debug", required=False, type=int, default=0)
args = ap.parse_args()

cwd = os.getcwd()

# initialize OpenCV's static fine grained saliency detector and
# compute the saliency map
saliency = cv.saliency.StaticSaliencyFineGrained_create()

W = 448
H = 448

def process(category):
    image_filenames = glob.glob('%s/*.png' % (category), recursive=True)
    image_filenames.extend(glob.glob('%s/*.jpg' % (category), recursive=True))
    # print("Start category: %s" % (category))
    for filename in image_filenames:
        # load the input image
        image = cv.imread(filename)
        h, w, c = image.shape

        # roi
        x_spacing = 20
        image = image[0:h, x_spacing:w-x_spacing] 
        h, w, c = image.shape

        if args.debug:
            print(w,h)
        (success, saliencyMap) = saliency.computeSaliency(image)

        '''
        # if we would like a *binary* map that we could process for contours,
        # compute convex hull's, extract bounding boxes, etc., we can
        # additionally threshold the saliency map
        if True:
            threshMap = cv.threshold(saliencyMap.astype("uint8"), 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        else:
            threshMap = cv.adaptiveThreshold(saliencyMap.astype("uint8"), 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 3, 5)
        
        M = cv.moments(threshMap.astype("uint8"))
        
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv.circle(image, (cX, cY), 10, (255, 0, 0), 1)

        if args.debug:
            cv.imshow("Thresh", threshMap)
        
        '''
        M = cv.moments(saliencyMap.astype("uint8"))
        
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv.circle(saliencyMap, (cX, cY), 5, (255, 255, 255), -1)

        cY = h / 2
        if cX < W / 2:
            cX = W / 2
        if cX > w - W / 2:
            cX = w - W / 2
        x1 = (int)(cX - W/2)
        y1 = (int)(cY - H/2)
        x2 = (int)(cX + W/2)
        y2 = (int)(cY + H/2)

        filename = filename.replace(args.images, 'img_roi').replace('.png', '.jpg')
        # print(filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        cv.imwrite(filename, image[y1:y2, x1:x2])

        if args.debug:
            # show the images
            cv.rectangle(saliencyMap, (x1,y1), (x2,y2), (255, 255, 255), 1)
            cv.imshow("Image", image)
            cv.imshow("Output", saliencyMap)

            key = cv.waitKey(0)
            if key == 27:
                break

    print("Finish category: %s" % (category))

def main():
    from multiprocessing.dummy import Pool as ThreadPool
    import multiprocessing

    category_folders = glob.glob('%s/*' % (args.images))
    cpu_n = multiprocessing.cpu_count()

    pool = ThreadPool(cpu_n)
    _ = pool.map(process, category_folders)

if __name__ == '__main__':
    main()
