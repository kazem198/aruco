import cv2
import numpy as np
import cv2.aruco as aruco
import os


def loadAugImages(path):
    myList = os.listdir(path)
    noOfMarkers = len(myList)
    augDics = {}
    for imgPath in myList:
        # print(imgPath)
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f'{path}/{imgPath}')
        augDics[key] = imgAug
        # print(augDics)
        return augDics


def findArucoMarkers(img, markerSize=4, totalMarkers=250, draw=True):

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
#     arucoDic = aruco.Dictionary_get(aruco.DICT_6X6_250)
    dictionary = aruco.getPredefinedDictionary(key)

    parameters = aruco.DetectorParameters()

    detector = aruco.ArucoDetector(dictionary, parameters)

    bbox, ids, rejected = detector.detectMarkers(imgGray)

    # print(ids, bbox)
    if draw:
        aruco.drawDetectedMarkers(img, bbox)

    return [bbox, ids]


def augmentAruco(bbox, id, img, imgAug, drawId=True):
    # print(bbox)

    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]
    # print(tl)

    h, w, c = imgAug.shape
    pts1 = np.array([tl, tr, br, bl])
    print(pts1)
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
    # cv2.fillPoly(img, [pts1.astype(int)], (0, 0, 0))
    imgOut = img + imgOut
    if drawId:
        cv2.putText(imgOut, str(id), (int(tl[0]), int(tl[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    return imgOut


def main():

    cap = cv2.VideoCapture(0)
    augDics = loadAugImages("Markers")

    while True:
        success, img = cap.read()
        arucoFound = findArucoMarkers(img)
        # print(len(arucoFound))

        if (len(arucoFound[0]) != 0):
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                if int(id) in augDics.keys():
                    img = augmentAruco(bbox, id, img, augDics[int(id)])

        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
