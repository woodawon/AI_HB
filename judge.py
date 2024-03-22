import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request as ur
import socket, threading
from collections import Counter
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Mecab
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr

ur.urlretrieve(
    "https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt",
    filename="ratings_total.txt",
)

total_data = pd.read_table("ratings_total.txt", names=["ratings", "reviews"])
"전체 답변 개수 :", len(total_data)  # 전체 답변 개수
# print(total_data[:5])

total_data["label"] = np.select([total_data.ratings > 3], [1], default=0)
# print(total_data[:5])

total_data["ratings"].nunique(), total_data["reviews"].nunique(), total_data[
    "label"
].nunique()

total_data.drop_duplicates(
    subset=["reviews"], inplace=True
)  # reviews 열에서 중복되는 내용이 있다면 중복 제거
"총 샘플의 수 :", len(total_data)

total_data.isnull().values.any()

train_data, test_data = train_test_split(total_data, test_size=0.25, random_state=42)
"훈련용 개수 :", len(train_data)
"테스트용 개수 :", len(test_data)

value_counts = train_data["label"].value_counts()
value_counts.plot(kind="bar")
# plt.title('Label Counts')
# plt.xlabel('Label')
# plt.ylabel('Count')
# plt.show()

train_data.groupby("label").size().reset_index(name="count")

# 한글과 공백을 제외하고 모두 제거
train_data["reviews"] = train_data["reviews"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
train_data["reviews"] = train_data["reviews"].apply(lambda x: x.replace(".", ""))
train_data["reviews"].replace("", np.nan, inplace=True)
# print(train_data.isnull().sum())

test_data.drop_duplicates(subset=["reviews"], inplace=True)  # 중복 제거
test_data["reviews"] = test_data["reviews"].str.replace(
    "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", ""
)  # 정규 표현식 수행
test_data["reviews"].replace("", np.nan, inplace=True)  # 공백은 Null 값으로 변경
test_data = test_data.dropna(how="any")  # Null 값 제거
# print('전처리 후 테스트용 샘플의 개수 :',len(test_data))

mecab_dic_path = "C:/mecab/mecab-ko-dic"  # 경로에 '/'를 사용
mecab = Mecab(mecab_dic_path)
# print(mecab.morphs('회사는 업무를 하는 곳이기 때문에 개인적인 일을 시키면 안된다고 생각합니다.'))

stopwords = [
    "도",
    "는",
    "다",
    "의",
    "가",
    "이",
    "은",
    "한",
    "에",
    "하",
    "고",
    "을",
    "를",
    "인",
    "듯",
    "과",
    "와",
    "네",
    "들",
    "듯",
    "지",
    "임",
    "게",
]

train_data["tokenized"] = train_data["reviews"].apply(mecab.morphs)
train_data["tokenized"] = train_data["tokenized"].apply(
    lambda x: [item for item in x if item not in stopwords]
)
test_data["tokenized"] = test_data["reviews"].apply(mecab.morphs)
test_data["tokenized"] = test_data["tokenized"].apply(
    lambda x: [item for item in x if item not in stopwords]
)

negative_words = np.hstack(train_data[train_data.label == 0]["tokenized"].values)
positive_words = np.hstack(train_data[train_data.label == 1]["tokenized"].values)

negative_word_count = Counter(negative_words)
# print(negative_word_count.most_common(20))

positive_word_count = Counter(positive_words)
# print(positive_word_count.most_common(20))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
text_len = train_data[train_data["label"] == 1]["tokenized"].map(lambda x: len(x))
ax1.hist(text_len, color="red")
ax1.set_title("Positive Reviews")
ax1.set_xlabel("length of samples")
ax1.set_ylabel("number of samples")
# print('긍정적인 답변의 평균 길이 :', np.mean(text_len))

text_len = train_data[train_data["label"] == 0]["tokenized"].map(lambda x: len(x))
ax2.hist(text_len, color="blue")
ax2.set_title("Negative Reviews")
fig.suptitle("Words in texts")
ax2.set_xlabel("length of samples")
ax2.set_ylabel("number of samples")
# print('부정적인 답변의 평균 길이 :', np.mean(text_len))
# plt.show()

X_train = train_data["tokenized"].values
y_train = train_data["label"].values
X_test = test_data["tokenized"].values
y_test = test_data["label"].values

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

threshold = 2
total_cnt = len(tokenizer.word_index)  # 단어의 수
rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if value < threshold:
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

# print('단어 집합(vocabulary)의 크기 :',total_cnt)
# print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
# print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
# print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

# 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거.
# 0번 패딩 토큰과 1번 OOV 토큰을 고려하여 +2
vocab_size = total_cnt - rare_cnt + 2
# print('단어 집합의 크기 :',vocab_size)

tokenizer = Tokenizer(vocab_size, oov_token="OOV")
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# print(X_train[:3])
# print(X_test[:3])

# print('답변의 최대 길이 :',max(len(review) for review in X_train))
# print('답변의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(review) for review in X_train], bins=50)
plt.xlabel("length of samples")
plt.ylabel("number of samples")
# plt.show()


def below_threshold_len(max_len, nested_list):
    count = 0
    for sentence in nested_list:
        if len(sentence) <= max_len:
            count = count + 1


#   print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))

max_len = 80
below_threshold_len(max_len, X_train)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(GRU(hidden_units))
model.add(Dense(1, activation="sigmoid"))

es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=4)
mc = ModelCheckpoint(
    "best_model.h5", monitor="val_acc", mode="max", verbose=1, save_best_only=True
)

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(
    X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2
)

loaded_model = load_model("best_model.h5")
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))


def sentiment_predict(new_sentence):
    new_sentence = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", new_sentence)
    new_sentence = mecab.morphs(new_sentence)
    new_sentence = [word for word in new_sentence if not word in stopwords]
    encoded = tokenizer.texts_to_sequences([new_sentence])
    pad_new = pad_sequences(encoded, maxlen=max_len)

    score = float(loaded_model.predict(pad_new))
    if score > 0.5:
        print("{:.2f}% 확률로 긍정적인 답변입니다.".format(score * 100))
        msg = score * 100
    else:
        print("{:.2f}% 확률로 부정적인 답변입니다.".format((1 - score) * 100))
        msg = (1 - score) * 100


r = sr.Recognizer()
with sr.Microphone() as source:
    audio = r.listen(source)

text = r.recognize_google(audio, language="ko")
sentiment_predict(text)


def binder(client_socket, addr):
    print("Connected by", addr)
    try:
        while True:
            data = client_socket.recv(4)
            length = int.from_bytes(data, "little")
            data = client_socket.recv(length)
            msg = data.decode()
            print("Received from", addr, msg)

            msg = "echo : " + msg
            data = msg.encode()
            length = len(data)
            client_socket.sendall(length.to_bytes(4, byteorder="little"))
            client_socket.sendall(data)
    except:
        print("except : ", addr)
    finally:
        client_socket.close()


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(("", 9999))
server_socket.listen()

try:
    while True:
        client_socket, addr = server_socket.accept()
        th = threading.Thread(target=binder, args=(client_socket, addr))
        th.start()
except:
    print("server")
finally:
    server_socket.close()
