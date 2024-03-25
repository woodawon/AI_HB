import csv
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# 비디오 캡처
video_capture = csv.VideoCapture(0)
#감정
f_detection = csv.CascadeClassifier("haarcascade_frontalface_default.xml")
emotion_classifier = load_model("_mini_XCEPTION.102-0.66.hdf5", compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surprising", "Neutral"]
#질문 & 답변
i = 0
questions = [ # 75점 기준 up & down
    "이 테스트는 모든 답변을 0~10 사이의 숫자로 답변해주시면 됩니다.(싫어요:0 / 좋아요:1) : ",
    "누구나 크고 작은 실수를 할 수 있다고 생각하시나요?(0~4 : 그렇지 않다 / 5 : 보통 / 6~10 : 그렇다) : ",
    "친한 친구와 함께 활동하는 것에 흥미를 느끼시나요?(0~4 : 그렇지 않다 / 5 : 보통 / 6~10 : 그렇다) : ",
    "동료가 맘에 들지 않는 행동을 했을 때 어떻게 대처하시나요?(0~4 : 약하게 대응 / 5 : 보통 / 6~10 : 강하게 대응) : ",
    "어떻게 자신의 감정을 표현하고 다른 사람들과 공유하나요?(0~4 : 은은한 / 5 : 보통 / 6~10 : 적극적) : ",
    "새로운 기술이나 개념을 배우는 것에 어떤 자세를 가지고 있나요?(0~4 : 신중한 / 5 : 보통 / 6~10 : 적극적) : ",
    "자신이 어떤 유형의 사람이라고 생각하나요?(0~4 : 서포터,수동적 / 5 : 보통 / 6~10 : 진취적,계획적) : ",
    "타인과의 갈등을 어떻게 해결하는 편이신가요?(0~4 : 조용히 덮기 / 5 : 보통 / 6~10 : 명확한 해결) : ",
    "어떤 방식으로 스트레스를 해소하거나 긍정적인 에너지를 얻나요?(0~4 : 내향적으로 / 5 : 보통 / 6~10 : 외향적으로) : ",
    "주변의 지인들이 당신을 어떤 사람으로 기억하는 편인가요?(0~4 : 묵묵하고 조용한 / 5 : 보통 / 6~10 : 밝고 긍정적) : ",
    "자신의 꿈과 목표를 달성하기 위해 어떤 방식으로 행동할 것 같나요?(0~4 : 묵묵히 꾸준하게 / 5 : 보통 / 6~10 : 용기내어 힘차게) : ",
    "어떤 가치가 당신의 선택과 행동을 주도하는 데 중요한 역할을 하나요?(0~4 : 실력, 스펙 / 5 : 보통 / 6~10 : 소통, 명예) : ",
    "우선순위를 결정할 때 어떤 기준을 사용하시나요?(0~4 : 쉬운 거 먼저 / 5 : 보통 / 6~10 : 어려운 거 먼저) : ",
    "어떤 종류의 영화가 당신의 기분을 좋게 만드나요?(0~4 : 잔잔하게 감상 가능한 / 5 : 보통 / 6~10 : 재밌고 자극적인) : ",
    "가족이나 지인들과 어떻게 시간을 보내는 것을 좋아하시나요?(0~4 : 소박하게 / 5 : 보통 / 6~10 : 화려하게) : ",
    "어떤 상황에서 가장 편안하게 느끼시나요?(0~4 : 조용한 / 5 : 보통 / 6~10 : 활기찬) : "
]

answers = []
result = 0

# 반복문
while True:
    res, frame = video_capture.read()
    image = csv.cvtColor(frame, csv.COLOR_BGR2GRAY)

    faces = f_detection.detectMultiScale(
        image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # empty image 만들기
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    resultCanvas = np.zeros((250, 300, 3), dtype="uint8")

    # 표정 탐지 기능 추가하기
    if len(faces) > 0:
        # 이미지 설정
        faced = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fx, fy, fw, fh) = faced
        roi = image[fy : fy + fh, fx : fx + fw]
        roi = csv.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # 감정 설정
        pred = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(pred)
        label = EMOTIONS[pred.argmax()]

        # 라벨링
        csv.putText(
            frame, label, (fx, fy - 10), csv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2
        )
        csv.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)

        #성격테스트 답변 입력
        answers.append(int(input(questions[i])))
        i += 1

        # 출력시키기
        for i, (emotion, prob) in enumerate(zip(EMOTIONS, pred)):
            text = "{}:{:.2f}".format(emotion, prob * 100)
            w = int(prob * 300)
            csv.rectangle(canvas, (7, (i * 35)), (w, (i * 35) + 35), (0, 0, 255), -1)
            csv.putText(
                canvas,
                text,
                (10, (i * 35) + 23),
                csv.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
            )

        # imshow
        csv.imshow("Feelings", frame)
        csv.imshow("Probabilities", canvas)

        # 마무리
        if csv.waitKey(1) & 0xFF == ord("g"):
            break

str = ""

for num in answers:
    result += num

if(result >= 75):
    str = "밝고 쾌활한 분위기메이커 대장이시군요!"
else:
    str = "조용히 빛나는 프로일잘러이시군요!"

csv.putText(
    resultCanvas,
    str,
    (50, 400),
    5,
    2,
    (254, 1, 15),
    3,
)
csv.imshow("Result", resultCanvas)


video_capture.release()
csv.destroyAllWindows()
