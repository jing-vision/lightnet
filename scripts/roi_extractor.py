import argparse
import cv2 as cv
import glob
import os
import pathlib
import darknet
import lightnet

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--images", required=False,
                    help="path to input image folder", default='img')
parser.add_argument("-d", "--debug", required=False, type=int, default=0)
parser.add_argument('--yolo_cfg')
parser.add_argument('--yolo_weights')

args = parser.parse_args()
args.images = pathlib.Path(args.images).as_posix()
cwd = os.getcwd()

# initialize OpenCV's static fine grained saliency detector and
# compute the saliency map
saliency = cv.saliency.StaticSaliencyFineGrained_create()

W = 448
H = 448

if args.yolo_cfg:
    yolo_net, yolo_meta = lightnet.load_network_meta(
        args.yolo_cfg, args.yolo_weights)


def process(category):
    image_filenames = glob.glob('%s/*.png' % (category), recursive=True)
    image_filenames.extend(glob.glob('%s/*.jpg' % (category), recursive=True))
    # print("Start category: %s" % (category))
    for filename in image_filenames:
        # load the input image
        filename = pathlib.Path(filename).as_posix()
        image = cv.imread(filename)
        h, w, c = image.shape

        # roi
        if False:
            x_spacing = 20
            image = image[0:h, x_spacing:w - x_spacing]
        h, w, c = image.shape

        if args.debug:
            print(w, h)

        x1, y1, x2, y2 = (0, 0, w, h)
        if args.yolo_cfg:
            full_im, _ = darknet.array_to_image(image)
            with inference_lock:
                results = lightnet.detect_from_memory(
                    yolo_net, yolo_meta, full_im, thresh=0.75, debug=False)
            if results:
                detection = results[0]
                x, y, w, h = detection[2][0],\
                    detection[2][1],\
                    detection[2][2],\
                    detection[2][3]
                if w > W and h > H:
                    x1, y1, x2, y2 = lightnet.convertBack(
                        float(x), float(y), float(w), float(h))
        else:
            (success, saliencyMap) = saliency.computeSaliency(image)
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
            x1 = int(cX - W / 2)
            y1 = int(cY - H / 2)
            x2 = int(cX + W / 2)
            y2 = int(cY + H / 2)

        filename = filename.replace(
            args.images, 'img_roi').replace('.png', '.jpg')
        # print(filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        cv.imwrite(filename, image[y1:y2, x1:x2])

        if args.debug:
            # show the images
            cv.rectangle(saliencyMap, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv.imshow("Image", image)
            cv.imshow("Output", saliencyMap)

            key = cv.waitKey(0)
            if key == 27:
                break

    print("Finish category: %s" % (category))


def main():
    global inference_lock
    from multiprocessing.dummy import Pool as ThreadPool
    import multiprocessing

    category_folders = glob.glob('%s/*' % (args.images))

    inference_lock = multiprocessing.Lock()
    cpu_n = multiprocessing.cpu_count()
    pool = ThreadPool(cpu_n)
    _ = pool.map(process, category_folders)

if __name__ == '__main__':
    main()
