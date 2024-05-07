import zipfile
import random
from matplotlib import pyplot as plt
import os
import io
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .classification_models import *
from .training_backend import *
from .detection_models import *
from .segmentation_models import *

from PIL import Image
import io
import base64


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),])

from django.shortcuts import render, HttpResponseRedirect, HttpResponse
from django.conf import settings

from .models import Project

def image(request):
    if request.method == 'POST':
        pjt_name = request.POST.get('project-name')
        desc = request.POST.get('project-description')
        dataset = request.FILES['dataset']
        pjt_type = request.POST.get('project-type')
        pjt = Project(name = pjt_name, description = desc, dataset = dataset, pjt_type=pjt_type)
        pjt.save()
        print(pjt.pk)
        print(pjt_type)
        if pjt_type == 'detection':
            return HttpResponseRedirect(f'detection/?pid={pjt.pk}&file={dataset}')
        elif pjt_type == 'classification':
            return HttpResponseRedirect(f'classification/?pid={pjt.pk}&file={dataset}')
        elif pjt_type == 'segmentation':
            return HttpResponseRedirect(f'segmentation/?pid={pjt.pk}&file={dataset}')
    return render(request,'image/image.html')

def detection(request):
    if request.method == 'GET':
        pid = request.GET.get('pid')
        
        with zipfile.ZipFile(os.path.join(settings.MEDIA_ROOT,'datasets_detect.zip'), 'r') as zip_ref:
            zip_ref.extractall("data")

        context = {'pid':pid}
        return render(request,'image/detection.html',context)
    return render(request,'image/detection.html')

def classify(request):
    if request.method == 'GET':
        pid = request.GET.get('pid')

        # unzipping files
        with zipfile.ZipFile(os.path.join(settings.MEDIA_ROOT,'dataset_classify.zip'), 'r') as zip_ref:
            zip_ref.extractall("data")

        # Getting file locations
        val_dir = os.path.join("data",'valid')
        val_data = ImageFolder(val_dir,transform=transform)

        # showing sample data
        plt.figure(figsize=(8, 2))
        fig, ax = plt.subplots(1,6)

        # collecting random images
        samples_idx = random.sample(range(len(val_data)), k=6)
            
        #iterating and getting plot
        for i, targ_sample in enumerate(samples_idx):
            targ_image, targ_label = val_data[targ_sample][0], val_data[targ_sample][1]

            targ_image_adjust = targ_image.permute(1, 2, 0)

        
            ax[i].imshow(targ_image_adjust)
            ax[i].axis("off")
            title = f"{val_data.classes[targ_label]}"
            ax[i].set_title(title, fontsize = 7)

            fig.suptitle("Sample input data")
        
        # Converting Images to IOBytes
        img_bytes = io.StringIO()
        plt.savefig(img_bytes, format='svg')
        img_bytes.seek(0)

        # Converting to Context
        img_bytes = img_bytes.getvalue()
        context = {'pid':pid, 'images':img_bytes}
        return render(request, 'image/classification.html', context)
    return render(request,'image/classification.html')

def segmentation(request):
    if request.method == 'GET':
        pid = request.GET.get('pid')
        
        with zipfile.ZipFile(os.path.join(settings.MEDIA_ROOT,'datasets_seg.zip'), 'r') as zip_ref:
            zip_ref.extractall("data")

        context = {'pid':pid}
        return render(request,'image/segmentation.html',context)
    return render(request,'image/segementation.html')

def training(request):
    operation = request.GET.get('type')
    if operation == 'classify':
        metrics = {}
        train_data = create_train_data()
        test_data = create_test_data()
        size = request.POST.get('size')
        classes = request.POST.get('classes')

        # For small models
        if size == 'small':
            priority = request.POST.get('priority')

            if priority == 'latency':
                metrics['MobileNet_V3_small'] = train_models(model=MobileNetV3_small(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10, save_name='MobileNet_V3_small')
                metrics['MNASet_1'] = train_models(model=mnasNet1(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10, save_name='MNASet_1')
                metrics['ShuffleNet_v2_X1'] = train_models(model=shuffnetv2_x0(output_classes=int(classes)), train_data=train_data, test_data=test_data, epochs=10, save_name='ShuffleNet_v2_X1')
            
            else:
                metrics['MobileNet_V3_small'] = train_models(model=MobileNetV3_small(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10, save_name='MobileNet_V3_small')
                metrics['MNASet_1'] = train_models(model=mnasNet1(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10, save_name='MNASet_1')
                metrics['ShuffleNet_v2_X1'] = train_models(model=shuffnetv2_x0(output_classes=int(classes)), train_data=train_data, test_data=test_data, epochs=10, save_name='ShuffleNet_v2_X1')
                metrics['DenseNet121'] = train_models(model=densenet121_model(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10, save_name='DenseNet121')
                metrics['EfficientNet_B0'] = train_models(model=effnetb0(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10, save_name='EfficientNet_B0')

        # For medium models
        elif size == 'medium':
            priority = request.POST.get('priority')

            if priority == 'latency':
                metrics['GoogleNet'] = train_models(model=googleNet(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10, save_name='GoogleNet')
                metrics['ResNet18'] = train_models(model=resnet18(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10, save_name='ResNet18')
            
            else:
                metrics['EfficientNet_B3'] = train_models(model=effnetb3(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10, save_name='EfficientNet_B3')
                metrics['DenseNet201'] = train_models(model=densenet201(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10, save_name='DenseNet201')
        
        # For large models
        else:
            priority = request.POST.get('priority')

            if priority == 'latency':
                metrics['Efficient_v2_small'] = train_models(model=googleNet(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10, save_name='Efficient_v2_small')
                metrics['ResNet50'] = train_models(model=resnet_50(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10,save_name='ResNet50')
                # metrics['Inception_v3'] = train_models(model=inception(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10, save_name='Inception_v3')
                
            
            else:
                metrics['EfficientNet_B5'] = train_models(model=effnetb5(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10, save_name='EfficientNet_B5')
                metrics['RegNet_Y_32GF'] = train_models(model=regnet32gf(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10, save_name='RegNet_Y_32GF')
                metrics['RegNet_Y_16GF'] = train_models(model=regnetY16gf(output_classes=int(classes)),train_data= train_data, test_data= test_data, epochs=10, save_name='RegNet_Y_16GF')

    elif operation == 'detection':
        if request.POST.get('size') == 'nano':
            model = yolonano()
            model.train(data="data/datasets/data.yaml",epochs=10)
            del model
        elif request.POST.get('size') == 'small':
            model = yolosmall()
            model.train(data="data/datasets/data.yaml",epochs=10)
            del model
        else:
            model = yolomed()
            model.train(data="data/datasets/data.yaml",epochs=10)
            del model

        image_file =  io.BytesIO(open('runs/detect/train/confusion_matrix.png', 'rb').read())
        image = Image.open(image_file)
        output = io.BytesIO()
        image.convert('RGB').save(output, 'PNG')
        image_file = io.BytesIO(open('runs/detect/train/confusion_matrix_normalized.png', 'rb').read())
        image1 = Image.open(image_file)
        output2 = io.BytesIO()
        image1.convert('RGB').save(output2, 'PNG')
        context = {"operation": operation,
                   "c_matrix_normalized": base64.b64encode(output.getvalue()).decode('utf-8'),
                   "c_matrix_test": base64.b64encode(output2.getvalue()).decode('utf-8')
                   }
        return render(request, 'image/results.html',context=context)
    else :
        if request.POST.get('size') == 'nano':
            model = segNano()
            model.train(data="data/datasets/data.yaml",epochs=10)
            del model
        elif request.POST.get('size') == 'small':
            model = segSmall()
            model.train(data="data/datasets/data.yaml",epochs=10)
            del model
        else:
            model = segMed()
            model.train(data="data/datasets/data.yaml",epochs=10)
            del model

        image_file =  io.BytesIO(open('runs/segment/train/labels_correlogram.jpg', 'rb').read())
        image = Image.open(image_file)
        output = io.BytesIO()
        image.convert('RGB').save(output, 'PNG')
        image_file = io.BytesIO(open('runs/segment/train/labels.jpg', 'rb').read())
        image1 = Image.open(image_file)
        output2 = io.BytesIO()
        image1.convert('RGB').save(output2, 'PNG')
        context = {"operation": operation,
                   "label_corr": base64.b64encode(output.getvalue()).decode('utf-8'),
                   "label": base64.b64encode(output2.getvalue()).decode('utf-8')
                   }
        return render(request, 'image/results.html',context=context)
    
        

    return render(request,'image/results.html', context= {"metrics":metrics, "operation": operation})

def download_models(request):
    if request.GET.get('type') == 'classify':
        name = request.GET.get('model')
        
        filename = name+'.pth'
        file_path = os.path.join(settings.MEDIA_ROOT, filename)

        f1 = open(file_path,'rb')
        response = HttpResponse(f1, content_type='application/force-download')
        response['Content-Disposition'] = "attachment; filename=%s" % filename
    elif request.GET.get('type') == 'detection':
        
        filename = 'best.pt'
        file_path = os.path.join('runs/detect/train/weights', filename)

        f1 = open(file_path,'rb')
        response = HttpResponse(f1, content_type='application/force-download')
        response['Content-Disposition'] = "attachment; filename=%s" % filename
    else:
        
        filename = 'best.pth'
        file_path = os.path.join('runs/segment/train/weights', filename)

        f1 = open(file_path,'rb')
        response = HttpResponse(f1, content_type='application/force-download')
        response['Content-Disposition'] = "attachment; filename=%s" % filename
    return response