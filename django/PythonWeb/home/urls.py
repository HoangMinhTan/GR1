from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('object_counting', views.object_counting),
    path('human_detect', views.human_detect),
    path('face_detect', views.face_detect),
    path('image_matching', views.image_matching)
]
