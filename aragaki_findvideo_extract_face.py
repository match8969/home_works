import cv2, os

img_last = None
front_no = 0
smile_no = 0
save_dir = "./out_faces"
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

# setting cascade
cascade_front_file = "haarcascade_frontalface_alt.xml"
cascade_smile_file = "haarcascade_smile.xml"
cascade_front = cv2.CascadeClassifier(cascade_front_file)
cascade_smile = cv2.CascadeClassifier(cascade_smile_file)

cap = cv2.VideoCapture("aragaki.mp4")
print("start!!")
while True:
    is_ok, frame = cap.read()
    if not is_ok:
        break

    # TODO: oraiginal is 640, 360. this is not needed.. maybe
    frame = cv2.resize(frame, (640, 360))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    img_b = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

    if not img_last is None:
        # TODO: add

        # frame_diff = cv2.absdiff(img_last, img_b)
        # cnts = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

        face_list_front = cascade_front.detectMultiScale(gray)
        face_list_smile = cascade_smile.detectMultiScale(gray)

        for x, y, w, h in face_list_front:
            # x, y, w, h = cv2.boundingRect(pt)
            # if w < 100 or w > 500:
            #     continue

            imgex = frame[y:y+h, x:x+w]
            outfile = save_dir + "/" + "front_"+str(front_no) + ".jpg"
            cv2.imwrite(outfile, imgex)

            front_no += 1

        for x, y, w, h in face_list_smile:
            # if w < 100 or w > 500:
            #     continue

            imgex = frame[y:y+h, x:x+w]
            outfile = save_dir + "/" + "smile_"+str(smile_no) + ".jpg"
            cv2.imwrite(outfile, imgex)

            smile_no += 1

    img_last = img_b

cap.release()
print("OK")