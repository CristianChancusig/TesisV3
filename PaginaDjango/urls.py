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
from django.contrib import admin
from django.urls import path,include
from django.conf import settings
from django.conf.urls.static import static
from PaginaDjango import views
from django.views.static import serve
from django.conf.urls import url

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
    url(r'^media/(?P<path>.*)$', serve,{'document_root':       settings.MEDIA_ROOT}),
    url(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}),
]
