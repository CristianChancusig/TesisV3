#from django.http import HttpResponse
#from django.template import Template, Context
#from django.template.loader import get_template
import imp
from re import A
from tokenize import String
from django.http import StreamingHttpResponse
from django.shortcuts import render
import cv2
from django.views.decorators import gzip
from matplotlib.style import context

from .forms import *
import resnet50

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import os
import shutil
import tensorflow as tf
from tensorflow import keras
import pathlib
import numpy as np

#from tensorflow.python.keras.models import load_model
#from tensorflow.python.keras.models import model_from_json

# --Cotraseñas AZURE--------
os.system("cls")
KEY = '6a68858c715a4dd2b4304010461a5082'
ENDPOINT = 'https://tesis-instv2.cognitiveservices.azure.com/'
print('-Listo para usar el servicio cognitivo de AZURE {} using key {}'.format(ENDPOINT, KEY))


cv_client = ComputerVisionClient(ENDPOINT, CognitiveServicesCredentials(KEY))
# ---------Variables-------
SelecR = ""
Obj1R = ""
Obj2R = ""
SelecREn = ""
Obj1En = ""
Obj2En = ""
listaOb = []
contImOb1 = 0
contImOb2 = 0
direccionDavid = "D:/David/ESPE/TESIS PROYECTO/PaginaDjango/PaginaDjango/static/Imagenes/Pr.jpg"
direccionCristian = "static/Imagenes/Pr.jpg"
dirCrisDatos = "Datos/"


def Red(request):
    # Obtener resultados
    TagsR = ""
    contexto = {"TagsI": TagsR, "SelecR1": SelecR,
                "Obj1": Obj1R, "Obj2": Obj2R}
    if request.method == "POST":
        if "BtnObjR" in request.POST:
            DatoRedp = DatosRed(request)
            contexto.update(DatoRedp)
    #contexto = {"IURL": ImURL, "TagsI": TagsR}
    return render(request, "paginaIn.html", contexto)

# ----Encender la cámara---------


def Activar1(request):
    encender = "on"
    contexto = {"OnOff1": encender, "SelecR1": SelecR,
                "Obj1": Obj1R, "Obj2": Obj2R}
    return render(request, "paginaIn.html", contexto)


def ActivarEn(request):
    encender = "on"
    contexto = {"OnOff1": encender, "SelecREn": SelecREn,
                "Obj1En": Obj1En, "Obj2En": Obj2En}
    # contexto.update(DatosEn)
    if request.method == "POST":
        if "btnCapEn" in request.POST:  # -Capturar fotos de entrenamiento
            camera = cv2.VideoCapture(0)
            _, img = camera.read()
            if (SelecREn == "Objeto 1"):
                global contImOb1
                contImOb1 = contImOb1+1
                dirDatos = f"{dirCrisDatos}{Obj1En}/{Obj1En}{contImOb1}.jpg"
                global contImOb2
                contImOb2 = 0
                print("----Se guardo la imagen en la direccion: " + dirDatos)
            else:
                contImOb2 = contImOb2+1
                dirDatos = f"{dirCrisDatos}{Obj2En}/{Obj2En}{contImOb2}.jpg"
                print("---Se guardo la imagen en la direccion 2: " + dirDatos)
                contImOb1 = 0
            cv2.imwrite(dirDatos, img)
            print("Se guardo la imagen")
    return render(request, "Entrenamiento.html", contexto)

# ----Apagar la cámara---------


def Apagar1(request):
    apagar = "off"
    contexto = {"OnOff1": apagar, "SelecR1": SelecR,
                "Obj1": Obj1R, "Obj2": Obj2R}
    # if request.method == "POST":
    #     if "BtnObjR" in request.POST:
    #         DatoRedp = DatosRed(request)
    #         contexto.update(DatoRedp)
    return render(request, "paginaIn.html", contexto)


def ApagarEn(request):
    apagar = "off"
    contexto = {"OnOff1": apagar, "SelecREn": SelecREn,
                "Obj1En": Obj1En, "Obj2En": Obj2En}
    if request.method == "POST":
        if "BtnObjR" in request.POST:
            DatoRedp = DatosRed(request)
            contexto.update(DatoRedp)
    return render(request, "Entrenamiento.html", contexto)

# ----Capturar la cámara---------


def Capturar(request):
    CapF = "Foto1"
    Foto = "Pr.jpg"
    tagsOb = []
    contexto = {"CapImg": CapF, "Foto": Foto}
    if request.method == "POST":
        if "BtnObjR" in request.POST:
            DatoRedp = DatosRed(request)
            contexto.update(DatoRedp)
    else:
        camera = cv2.VideoCapture(0)
        _, img = camera.read()
        cv2.imwrite(direccionCristian, img)
        filepath = direccionCristian
        with open(filepath, mode='rb') as image_stream:
            # Call API with remote image
            tags_result_remote = cv_client.tag_image_in_stream(
                image_stream, language="es")
            # Print results with confidence score
            print("----------TAGS:-------- ")
            if (len(tags_result_remote.tags) == 0):
                print("No tags detected.")
            else:
                for tag in tags_result_remote.tags:
                    tagsOb.append(tag.name)
                    global listaOb
                    listaOb = tagsOb
                    print(format(tag.name))
        ListaObjeto = {"ListObj": tagsOb}
        contexto.update(ListaObjeto)
    return render(request, "paginaIn.html", contexto)

# ----PRUEBA---


def idenObjeto(request):
    filepath = direccionCristian
    with open(filepath, mode='rb') as image_stream:
        # Call API with remote image
        tags_result_remote = cv_client.tag_image_in_stream(
            image_stream, language="es")
        # Print results with confidence score
        print("----------TAGS:-------- ")
        if (len(tags_result_remote.tags) == 0):
            print("No tags detected.")
        else:
            for tag in tags_result_remote.tags:
                print(format(tag.name))
                if (tag.name == Obj1R):
                    print("Primera opción es " + Obj1R)
                elif(tag.name == Obj2R):
                    print("Segunda opción es " + Obj2R)
    contexto = {"ListObj": tags_result_remote.tags.name}
    return render(request, "Entrenamiento.html", contexto)

# ---Obtención de etiquetas en la red principal


def DatosRed(request):
    form = dato_RedOb(request.POST)
    if form.is_valid():
        global SelecR
        SelecR = request.POST['slcObj']
        if (SelecR == "Objeto 1"):
            global Obj1R
            Obj1R = request.POST['listObjR']
        else:
            global Obj2R
            Obj2R = request.POST['listObjR']
    else:
        print("No es valido el formulario")
    contexto = {"SelecR1": SelecR, "Obj1": Obj1R, "Obj2": Obj2R, }
    ListaObjeto = {"ListObj": listaOb}
    contexto.update(ListaObjeto)
    return contexto

# ---Obtencion de etiquetas en el entrenamiento


def DatosEntr(request):
    form = dato_Tag(request.POST)
    if form.is_valid():
        global SelecREn
        SelecREn = request.POST['slcObTag']
        if (SelecREn == "Objeto 1"):
            global Obj1En
            Obj1En = request.POST['txtTag']
        else:
            global Obj2En
            Obj2En = request.POST['txtTag']
    else:
        print("No es valido el formulario")
    contexto = {"SelecREn": SelecREn, "Obj1En": Obj1En, "Obj2En": Obj2En, }
    # ListaObjeto = {"ListObj": listaOb}
    # contexto.update(ListaObjeto)
    return contexto
# ----------Pagina Entrenamiento----------


def Train(request):
    TagsR = ""
    contexto = {"TagsI": TagsR, "SelecREn": SelecREn,
                "Obj1En": Obj1En, "Obj2En": Obj2En}
    if request.method == "POST":
        if "btnTags" in request.POST:
            Eform = dato_Tag(request.POST)
            if Eform.is_valid():
                fileCarp = request.POST['txtTag']
                DatosEn = DatosEntr(request)
                contexto.update(DatosEn)
                if(os.path.exists('Datos/'+fileCarp)):
                    print("Ya existe esa carpeta")
                else:
                    os.mkdir('Datos/'+fileCarp)
            else:
                print("No es valido el formulario")
        elif "btnNuevoEntr" in request.POST:
            shutil.rmtree('Datos')
            os.mkdir('Datos')
    return render(request, "Entrenamiento.html", contexto)

# ---- Función Entrenamiento-----------


def Entrenando(request):
    print("Empieza entrenamiento")
    contexto = {"SelecREn": SelecREn,
                "Obj1En": Obj1En, "Obj2En": Obj2En}
    resnet50.Entrenar()
    global modelo_final, train_ds
    modelo_final, train_ds = CargarModelo(
        "modelo.model", "Datos")
    print("---------Modelo listo para usar-------")
    return render(request, "Entrenamiento.html", contexto)


# ----Carga de modelo------
def CargarModelo(modelo, data_dir):
    print("-------Cargando modelo----------")
    new_model = tf.keras.models.load_model(modelo)

    data_dir = pathlib.Path(data_dir)

    img_height, img_width = 180, 180
    batch_size = 32
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size)
    print("-------Modelo cargado----------")

    return new_model, train_ds

# ----Predicción con red local----


def Prediccion2(request):
    print("--------Predicción Prueba--------")
    img_height, img_width = 180, 180
    impred = input("Ingrese la imagen: ")
    path = f"static/Imagenes/{impred}.jpg"
    image = cv2.imread(path)
    image_resized = cv2.resize(image, (img_height, img_width))
    image = np.expand_dims(image_resized, axis=0)
    class_name = train_ds.class_names

    pred = modelo_final.predict(image)
    output_class = class_name[np.argmax(pred)]
    print("La clase de predicción es", output_class)
    return render(request, "Entrenamiento.html")

# --Obtener objeto de red-------


def getObjecto(request):
    cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    xlimite = 180
    cont = 0

    if not cap.isOpened():
        raise IOError('Error en la camara')

    while True:
        ret, frame = cap.read()
        ret2, frame2 = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
        color = (0, 255, 0)
        mensaje = "Vacio"
        pts_area = np.array([[150, 100], [500, 100], [500, 400], [
                            150, 400]])  # área de detección
    # Área a detectar
        img_aux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
        img_aux = cv2.drawContours(img_aux, [pts_area], -1, (255), -1)
        img_det = cv2.bitwise_and(gray, gray, mask=img_aux)
        fgmask = fgbg.apply(img_det)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.dilate(fgmask, None, iterations=2)
    # Limite para capturar foto
        # Límite en donde va a tomar la foto
        cv2.line(frame, (xlimite, 100), (xlimite, 400), (155, 20, 150), 2)
    # ----------------------------
        cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
        for cnt in cnts:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                color = (0, 0, 255)
                mensaje = "Hay un objeto"
                # --------
                if x <= xlimite and cont <= 2:
                    if cont == 2:
                        cv2.imwrite(
                            'static/Imagenes/ifigura1.jpg', frame2)
                            

                        # PARA IMAGEN EN EL PC
                        filepath = 'static/Imagenes/ifigura1.jpg'
                        with open(filepath, mode='rb') as image_stream:
                            # Call API with remote image
                            tags_result_remote = cv_client.tag_image_in_stream(
                                image_stream, language="es")
                            # Print results with confidence score
                            print("----------TAGS:-------- ")
                            if (len(tags_result_remote.tags) == 0):
                                print("No tags detected.")
                            else:
                                for tag in tags_result_remote.tags:
                                    print("'{}' with confidence {:.2f}%".format(
                                        tag.name, tag.confidence * 100))
                                    if (tag.name == Obj1R):
                                        print("Primera opción es " + Obj1R)
                                    elif(tag.name == Obj2R):
                                        print("Segunda opción es " + Obj2R)
                    cv2.putText(frame, "Captura", (200, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cont = cont + 1
                    # print(cont)
                # --------
        # Dibujamos el contorno de los puntos
        cv2.drawContours(frame, [pts_area], -1, color, 2)
        cv2.putText(frame, mensaje, (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if x > 300 and cont >= 2:  # reseteo para tomar una foto
            cont = 0
        cv2.imshow('Camara', frame)
        c = cv2.waitKey(1)
        if c == 27:
            break
    return render(request, "paginaIn.html")


def pruebaCam(request):
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.9, fy=0.9,
                           interpolation=cv2.INTER_AREA)
        cv2.imshow('Input', frame)

        c = cv2.waitKey(1)
        if c == 113:
            #cv2.imwrite('D:\Cristian\Docu_Tesis\Prueba1\ImagenesP\ifigura1.jpg', frame)
            print("Capturado")

        if c == 27:
            break
    print('Final')
    # cv2.imwrite('D:\Cristian\Docu_Tesis\Prueba1\ImagenesP\ifigura1.jpg',frame)

    cap.release()
    cv2.destroyAllWindows()
    return render(request, "paginaIn.html")
# ----- Red pruebas---


def get_frame():
    camera = cv2.VideoCapture(0)
    while True:
        _, img = camera.read()
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        imgencode = cv2.imencode('.jpg', img)[1]
        stringData = imgencode.tostring()
        yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
    del(camera)


@gzip.gzip_page
def dynamic_stream(request, stream_path="video"):
    try:
        return StreamingHttpResponse(get_frame(), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        return "error"
