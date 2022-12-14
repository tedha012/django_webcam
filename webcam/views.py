from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from webcam.camara import FaceDetect


def index(request):
    return render(request, "webcam/index.html")


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


def predict(request):
    get_text = FaceDetect()
    pred = get_text.get_text()
    gender_pred = pred[0][0]
    age_pred = pred[0][1]
    return render(
        request,
        "webcam/predict.html",
        {
            "gender": gender_pred,
            "age": age_pred,
        },
    )


def webcam_feed(request):
    return StreamingHttpResponse(
        gen(FaceDetect()), content_type="multipart/x-mixed-replace; boundary=frame"
    )
