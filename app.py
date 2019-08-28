from source.nn import GestureRecognizer
import cv2

class APP():
    def __init__(self):
        self.classifier = GestureRecognizer()
        self.y = 60
        self.x = 60
        self.h = 450
        self. w = 450

    
    def run(self):

        while(True):
            ret, frame = cap.read()
            ret, frame2 = cap.read()\
            crop = frame2[y:y+h, x:x+w].copy()\n",
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)\n",
            cv2.putText(frame, str(number), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, color_black, 2)\n",
            cv2.rectangle(frame, (60,450), (450,60), color_red, thickness=2, lineType=8, shift=0)\n",
            cv2.imshow('Video', frame)\n",
            cv2.imshow('Video2',crop)\n",
            frame = None
            image_class = self.classifier.predict(frame)
            print(image_class)

        return


# код ниже запустит приложение
if __name__ == '__main__':
    try:
        myApp = APP()
        myApp.run()
    except KeyboardInterrupt:
        print('\n\nApp was stopped.\n')
    
