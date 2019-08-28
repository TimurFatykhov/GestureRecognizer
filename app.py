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
        color_green = (0, 255, 0)
        color_black = (0, 0, 0)
        cam = cv2.VideoCapture(0)
        while True:
            _, frame = cam.read()
            height, width, channels = frame.shape
            frame = cv2.resize(frame, (640, 360))
            frame = cv2.flip(frame, 1)

            crop = frame[20:320, 160:480].copy()

            number = self.classifier.predict(frame)
            cv2.putText(frame, str(number), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 2, color_green, 2)
            cv2.rectangle(frame, (160,20), (480,320), color_green, thickness=2, lineType=8, shift=0)

            cv2.imshow('my webcam', frame)
            cv2.imshow('crop', crop)
            if cv2.waitKey(1) == 27: 
                break  # esc to quit
        cv2.destroyAllWindows()
        return


# код ниже запустит приложение
if __name__ == '__main__':
    try:
        myApp = APP()
        # myApp.run()
    except KeyboardInterrupt:
        print('\n\nApp was stopped.\n')
