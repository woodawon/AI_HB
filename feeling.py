import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import socket

video_capture = cv2.VideoCapture(0)
f_detection = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
emotion_classifier = load_model("_mini_XCEPTION.102-0.66.hdf5", compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surprising", "Neutral"]

while True:
    res, frame = video_capture.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = f_detection.detectMultiScale(
        image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # empty image 만들기
    canvas = np.zeros((250, 300, 3), dtype="uint8")

    # 표정 탐지 기능 추가하기
    if len(faces) > 0:
        # 이미지 설정
        faced = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fx, fy, fw, fh) = faced
        roi = image[fy : fy + fh, fx : fx + fw]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # 감정 설정
        pred = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(pred)
        label = EMOTIONS[pred.argmax()]

        # 라벨링
        cv2.putText(
            frame, label, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2
        )
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)

        # 출력시키기
        for i, (emotion, prob) in enumerate(zip(EMOTIONS, pred)):
            text = "{}:{:.2f}".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35)), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(
                canvas,
                text,
                (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
            )

        cv2.putText(
            frame,
            "기업의 ESG 문화를 더욱 번영시키려면 모두가 노력해야 한다.",
            (50, 400),
            5,
            2,
            (254, 1, 15),
            3,
        )

        # Display
        ## Display image ("Feelings")
        ## Display Probabilities of emotion

        # imshow
        cv2.imshow("Feelings", frame)
        cv2.imshow("Probabilities", canvas)

        # 마무리
        if cv2.waitKey(1) & 0xFF == ord("g"):
            break


video_capture.release()
cv2.destroyAllWindows()
