"""PaginaDjango URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
import imp
from django.contrib import admin
from django.urls import path
from PaginaDjango import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('Red/', views.Red, name="Red"),
    path('Train/', views.Train, name="Train"),
    path('Lista/', views.idenObjeto, name="Lista"),
    path('videostream/', views.dynamic_stream, name="videostream"),
    path('Activar1/', views.Activar1, name="Activar1"),
    path('ActivarEn/', views.ActivarEn, name="ActivarEn"),
    path('Apagar1/', views.Apagar1, name="Apagar1"),
    path('ApagarEn/', views.ApagarEn, name="ApagarEn"),
    path('Capturar/', views.Capturar, name="Capturar"),
    path('PruebaCamara/', views.getObjecto, name="PruebaCamara"),
    path('Entrenando/', views.Entrenando, name="Entrenando"),
    path('Prediccion2/', views.Prediccion2, name="Prediccion2"),
]
