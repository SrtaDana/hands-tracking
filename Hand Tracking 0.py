#Initial Code Retrieved from: https://www.makeuseof.com/python-hand-tracking-opencv/
import cv2
import mediapipe as mp
import imutils


# Declaring MediaPipe objects
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


# Processing the input image
def process_image(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(gray_image)
    # print(results.multi_hand_landmarks)
    return results


# Drawing landmark connections
def draw_hand_connections(img, results):
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape

                # Finding the coordinates of each landmark
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Printing each landmark ID and coordinates
                #print(id, cx, cy)
                if id==8:
                    if cx<250:
                        print("Su dedo indice está a la derecha")  
                    
                    elif cx> 250:
                        print("Su dedo indice está a la izquierda")  


                # Creating a circle around each landmark
                cv2.circle(img, (cx, cy), 10, (0, 255, 0),
                           cv2.FILLED)
                # Drawing the landmark connections
                mpDraw.draw_landmarks(img, handLms,
                                      mpHands.HAND_CONNECTIONS)

        return img


def main():
    cap = cv2.VideoCapture(0)
    while True:
        success, image = cap.read()
        image = imutils.resize(image, width=500, height=500)
        results = process_image(image)
        draw_hand_connections(image, results)

        cv2.imshow("Hand tracker", image)

        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
