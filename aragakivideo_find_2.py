import cv2, os, copy
from sklearn.externals import joblib

clf = joblib.load("aragaki_front.pkl")
output_dir = "./aragaki_bestshot_front"
img_last = None
aragaki_th = 1
count = 0
frame_count = 0

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# setting cascade
cascade_front_file = "haarcascade_frontalface_alt.xml"
cascade_front = cv2.CascadeClassifier(cascade_front_file)


cap = cv2.VideoCapture("aragaki.mp4")

# test
fmt = cv2.VideoWriter_fourcc('m','p','4','v')
fps = 20.0
size = (640, 360)
writer = cv2.VideoWriter('output.m4v', fmt, fps, size)

print("start!!!!")
while True:
    is_ok, frame = cap.read()
    if not is_ok:
        break
    frame = cv2.resize(frame, (640, 360))
    frame2 = copy.copy(frame)
    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (15, 15), 0)
    img_b = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    if not img_last is None:
        # frame_diff = cv2.absdiff(img_last, img_b)
        # cnts = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        # aragaki_count = 1
        face_list_front = cascade_front.detectMultiScale(gray)

        for x, y, w, h in face_list_front:

            # if w < 100 or x > 500:
            #     continue
            imgex = frame[y:y+h, x:x+w]
            imagex = cv2.resize(imgex, (64, 32))
            image_data = imagex.reshape(-1, )
            pred_y = clf.predict([image_data])
            if pred_y[0] == 1:
                # aragaki_count += 1
                rectangle_frame = cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)
                writer.write(rectangle_frame)
            else:
                writer.write(frame)
        # if aragaki_count == aragaki_th:
        #     fname = output_dir + "/aragaki" + str(count) + ".jpg"
        #     cv2.imwrite(fname, frame)
        #     count += 1

    cv2.imshow('ARAGAKI!!', frame2)
    if cv2.waitKey(1) == 13:
        break
    img_last = img_b

cap.release()
writer.release()
cv2.destroyAllWindows()
print("OK", count, "/", frame_count)