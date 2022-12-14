from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("predict", views.predict, name="predict"),
    path("webcam_feed", views.webcam_feed, name="webcam_feed"),
]
