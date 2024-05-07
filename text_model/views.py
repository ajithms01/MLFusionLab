from django.shortcuts import render, HttpResponseRedirect, HttpResponse
from django.conf import settings

from .models import Project
from .sentiment_backend import *
import shutil
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def text(request):
    if request.method == 'POST':
        pjt_name = request.POST.get('project-name')
        desc = request.POST.get('project-description')
        csv_file = request.FILES['csv_file']
        pjt = Project(name = pjt_name, description = desc, csv_file = csv_file)
        pjt.save()
        return HttpResponseRedirect(f'sentiment/?pid={pjt.pk}')
    return render(request,'text/text.html')

def sentiment(request):
    if request.method == 'GET':
        pid = request.GET.get('pid')
        file_path = Project.objects.filter(pk=pid).first().csv_file
        df = pd.read_csv(file_path)
        fields = list(df.columns)
        df = df.to_html(max_rows=6,classes=['table px-4 border border-info-subtle rounded'],justify='center')
        context = {'df_table': df,'columns': fields,'pid':pid}
        return render(request, 'text/sentiment.html', context)
    return render(request, 'text/sentiment.html')

def training(request):
    if request.method == 'POST':
        project_id = request.GET.get('pid')
        target = request.POST.get('target')
        text_data = request.POST.get('text_data')
        num_labels = int(request.POST.get('num_labels'))
        filepath = Project.objects.filter(pk=project_id).first().csv_file
        print(filepath)
        df = pd.read_csv(filepath)
        lab = LabelEncoder()
        df[target] = lab.fit_transform(df[target])
        df['data_type'] = ['not_set']*df.shape[0]
        xtrain, xval, ytrain, yval = csv_split(index=df.index.values,category=df[target].values, category_values=df[target].values)
        df.loc[xtrain, 'data_type'] = 'train'
        df.loc[xval, 'data_type'] = 'val'

        dataset_train, dataset_val = dataset_creation(df=df,target=target,text_data=text_data)

        val_loss, val_f1 = training_model(dataset_train,dataset_val,num_labels=num_labels)

        return render(request,"text/results.html",context={"val_loss":val_loss,"f1":val_f1})
    
def download_models(request):
    origin = 'BERT.model'
    target = 'models/'
    
    shutil.move(origin, target+origin)

    file_path = os.path.join(settings.MEDIA_ROOT,origin)
    
    f1 = open(file_path,'rb')
    response = HttpResponse(f1, content_type='application/force-download')
    response['Content-Disposition'] = "attachment; filename=%s" % origin
    
    return response




