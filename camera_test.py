import cv2

cap = cv2.VideoCapture(0)

print("Camera opened:", cap.isOpened())

while True:
    success, frame = cap.read()

    if not success:
        print("Cannot read camera")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()