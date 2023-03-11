import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from time import sleep
from pynput.keyboard import Controller


cap = cv2.VideoCapture(0)

detector = HandDetector(detectionCon=0.8, maxHands=2)

finalText=""
keyword = Controller()
buttonList = []
keys = [["Q","W","E","R","T","Y","U","I","O","P"],
        ["A","S","D","F","G","H","J","K","L",";"],
        ["Z","X","C","V","B","N","M",",",".","/"]]




def drawAll(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)

    for button in buttonList:
        x, y = button.pos


        cvzone.cornerRect(imgNew,(button.pos[0],button.pos[1],button.size[0],button.size[1]
                                  ),20, rt=0)
        cv2.rectangle(imgNew, button.pos, (x + button.size[0], y + button.size[1]), (255, 0, 255), cv2.FILLED)

        cv2.putText(imgNew,button.text,(x + 40, y+60),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)


        out = img.copy()
        alpha =0.5
        mask = imgNew.astype(bool)
        out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    return out



class Button():
    def __init__(self, pos, text, size=[85,85]):
        self.pos = pos
        self.size = size
        self.text = text



for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))


while True:
    success, img = cap.read()

    # For left to left and right to right
    img = cv2.flip(img, 1)

    # Find the hand and its landmarks
    lmList, bboxInfo = detector.findHands(img)  # with draw


    img = drawAll(img, buttonList)

    if lmList:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            if x < lmList[0]['lmList'][8][0] < x+w and y < lmList[0]['lmList'][8][1] < y+h:
                cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 0), cv2.FILLED)
                cv2.putText(img, button.text, (x + 15, y + 70),
                            cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

                l, _, _ = detector.findDistance(lmList[0]['lmList'][8][:2], lmList[0]['lmList'][12][:2], img)

                if l < 50:
                    keyword.press(button.text)
                    cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 15, y + 70),
                                cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)
                    finalText += button.text

                    sleep(0.3)

    cv2.rectangle(img, (50,350), (1040, 450), (175, 0, 175), cv2.FILLED)

    cv2.putText(img, finalText, (60, 425),
                cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
