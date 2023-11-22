import numpy as np
import cv2 as cv

def rect_crop(im, rect):
    # crop rectangle out of image
    crop = im.copy()
    x, y, w, h = rect
    return crop[y:y+h, x:x+w]

def square_crop(im):
    # crop image into square
    h1, w1 = im.shape[:2]
    h2 = w2 = min(h1, w1)

    left = int(np.ceil((w1 - w2) / 2))
    right = w1 - int(np.floor((w1 - w2) / 2))

    top = int(np.ceil((h1 - h2) / 2))
    bottom = h1 - int(np.floor((h1 - h2) / 2))

    return im[top:bottom, left:right]

class MikeWazowskiDetector:
    def __init__(
            self, fac=.1, 
            template='./iris.png',
            template_factors=(.75, 1, 1.5, 2, 2.5)
    ):
        # images get downscaled for template matching
        self.fac = fac
        self.im_template = self.load_image(template)
        # creating different scales for the iris template
        self.templates = [self.prep_image(self.im_template, fac) for fac in (w*fac for w in template_factors)]

    @staticmethod
    def load_image(filepath):
        return cv.cvtColor(cv.imread(filepath), cv.COLOR_BGR2RGB)

    @staticmethod
    def prep_image(im, fac):
        # low-res grayscale for templating matching
        low_res = cv.resize(cv.cvtColor(im, cv.COLOR_RGB2GRAY), (0,0), fx=fac, fy=fac, interpolation=cv.INTER_NEAREST)
        return low_res
    
    @staticmethod
    def resize_bbox(iris_bbox, im_shape):
        x, y, w, h = iris_bbox
        # magic scaling numbers to find bounding box of mike given
        # the bounding box of his iris
        x = x+w/2 - 2.5*w
        w = 5*w
        y = y+h/2 - 2*h
        h = 5*h
        # clipping indices        
        x = max(x, 0)
        w = min(x+w, im_shape[1]) - x
        y = max(y, 0)
        h = min(y+h, im_shape[0]) - y

        return [int(e) for e in (x, y, w, h)]
       

    @staticmethod
    def contains_big_circle(im):
        # checking if the found iris contains a circle using Hough transform
        # template matching gives us a lot of false positives, escpecially if
        # the iris is not in the image, hence we check if there is a proiment
        # circle in the bounding box
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (11,11), 3)

        circles = cv.HoughCircles(
            gray, cv.HOUGH_GRADIENT,
            dp=1, minDist=20, param1=50, param2=30, 
            minRadius=0, maxRadius=0 
        )
        # if the most promiment circle is at least 1/3 of the bbox we confirm our match
        if circles is not None:
            return circles[0,0,-1] > im.shape[0]/3
        return False

        
    def match_templates(self, low_res, im):
        # multi-scale template matching on the low res imagery
        results = [cv.matchTemplate(low_res, t, cv.TM_CCOEFF) for t in self.templates]
        # normalize results from tempate matching
        results_norm = [np.clip(res / (res.max() - res.min()), 0, 1) for res in results]
        # we check the most confident match
        i = np.argmax([r.max() for r in results_norm])
        if results_norm[i].max() > .55 :
            # find bounding box
            x, y = cv.minMaxLoc(results[i])[-1]
            h, w = self.templates[i].shape
            # rescale to original image size
            x, y, w, h = [int(e/self.fac) for e in (x, y, w, h)]
            # match must be somewhere in the middle of the image (assmuing 1080p images)
            if (y > 600) or (y < 200) or (x < 250) or (x > 1600) : return None
            # check for circles
            if self.contains_big_circle(rect_crop(im, [x, y, w, h ])):
                return self.resize_bbox((x, y, w, h), im.shape)
        
        return None
    
    def find_mike(self, im, draw_bbox=True):
        # entry point for detecting mike in image
        low_res = self.prep_image(im, self.fac)
        mike = self.match_templates(low_res, im)

        if mike and draw_bbox:
            x, y, w, h = mike
            cv.rectangle(im, (x, y), (x+w, y+h), 255, 10)

        return mike


def main(in_path, out_path):
    mike_detector = MikeWazowskiDetector()

    files = glob(os.path.join(in_path, '*.jpg'))

    mike_count = 0
    for i, file in tqdm(enumerate(files), total=len(files)):
        im = mike_detector.load_image(file)
        mike = mike_detector.find_mike(im, draw_bbox=False)

        if mike:
            mike_count += 1
            cv.imwrite(
                os.path.join(out_path, f'{mike_count}-{i+1:06d}.jpg'), 
                cv.cvtColor(cv.resize(square_crop(rect_crop(im, mike)), (128,128), interpolation=cv.INTER_AREA), cv.COLOR_RGB2BGR)
            )


if __name__ == '__main__':
    from glob import glob
    from tqdm import tqdm
    import os

    in_path = './buffer/'
    out_path = './out/'

    main(in_path, out_path)