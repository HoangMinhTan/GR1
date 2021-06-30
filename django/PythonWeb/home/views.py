from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
def index(request):
    return render(request, 'pages/home.html')

def object_counting(request):
    return render(request, 'pages/object_counting.html')

def human_detect(request):
    return render(request, 'pages/object_counting.html')

def face_detect(request):
    return render(request, 'pages/object_counting.html')

def image_matching(request):
    return render(request, 'pages/object_counting.html')
