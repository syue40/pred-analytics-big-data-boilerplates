from django.shortcuts import render, HttpResponseRedirect
from django.http import Http404
from django.urls import reverse
from django.views.generic import TemplateView
import pickle
import sklearn # You must perform a pip install.
import pandas as pd
import statsmodels.api as sm

def homePageView(request):
    return render(request, 'home.html', {
        'labels':['radius_mean','texture_mean','perimeter_mean',
                    'area_mean', 'smoothness_mean', 'area_se', 
                    'smoothness_worst', 'concave_points_worst','symmetry_worst']
        })
# pages/urls.py

def homePost(request):
    # Create variable to store choice that is recognized through entire function.
    radius_mean = 0.0
    texture_mean = 0.0
    perimeter_mean = 0.0
    area_mean = 0.0
    smoothness_mean = 0.0
    area_se = 0.0
    smoothness_worst = 0.0
    concave_points_worst = 0.0
    symmetry_worst = 0.0
    
    try:
        # Extract value from request object by control name.
        choices = request.POST
        print(request.POST)
        radius_mean = float(request.POST.get('radius_mean'))
        texture_mean = float(request.POST.get('texture_mean'))
        perimeter_mean = float(request.POST.get('perimeter_mean'))
        area_mean = float(request.POST.get('area_mean'))
        smoothness_mean = float(request.POST.get('smoothness_mean'))
        area_se = float(request.POST.get('area_se'))
        smoothness_worst = float(request.POST.get('smoothness_worst'))
        concave_points_worst = float(request.POST.get('concave_points_worst'))
        symmetry_worst = float(request.POST.get('symmetry_worst'))
        # Crude debugging effort.
        print("*** Your details: " + str(choices))
        # choice = int(leChoix)
        # gmat = float(gmatStr)
        # Enters 'except' block if integer cannot be created.
    except:
        return render(request, 'home.html', {
        'errorMessage':'Please Check that your data is in the correct format.',
        'labels': ['radius_mean','texture_mean','perimeter_mean',
                                           'area_mean', 'smoothness_mean', 'area_se', 
                                           'smoothness_worst', 'concave_points_worst','symmetry_worst']})
    else:
    # Always return an HttpResponseRedirect after successfully dealing
    # with POST data. This prevents data from being posted twice if a
    # user hits the Back button.
        return HttpResponseRedirect(reverse('results', kwargs={'radius_mean':radius_mean,
    'texture_mean':texture_mean, 'perimeter_mean': perimeter_mean, 'area_mean': area_mean,
    'smoothness_mean': smoothness_mean, 'area_se': area_se, 'smoothness_worst': smoothness_worst, 
    'concave_points_worst': concave_points_worst, 'symmetry_worst': symmetry_worst},))
    

def results(request, radius_mean,
    texture_mean,
    perimeter_mean,
    area_mean,
    smoothness_mean,
    area_se,
    smoothness_worst,
    concave_points_worst,
    symmetry_worst,):
    print("*** Inside reults()")
    # load saved model
    scores = {} # scores is an empty dict already
    with open('logRegModel1.pkl' , 'rb') as file:
        loadedModel = pickle.load(file)
        file.close()
        
    # Create a single prediction.
    singleSampleDf = pd.DataFrame(columns=['radius_mean','texture_mean','perimeter_mean',
                                           'area_mean', 'smoothness_mean', 'area_se', 
                                           'smoothness_worst', 'concave points_worst','symmetry_worst'])
    radius_mean = float(radius_mean)
    texture_mean = float(texture_mean)
    perimeter_mean = float(perimeter_mean)
    area_mean = float(area_mean)
    smoothness_mean = float(smoothness_mean)
    area_se = float(area_se)
    smoothness_worst = float(smoothness_worst)
    concave_points_worst = float(concave_points_worst)
    symmetry_worst = float(symmetry_worst)
    
    
    
    choices = {'radius_mean':radius_mean,
    'texture_mean':texture_mean, 'perimeter_mean': perimeter_mean, 'area_mean': area_mean,
    'smoothness_mean': smoothness_mean, 'area_se': area_se, 'smoothness_worst': smoothness_worst, 
    'concave points_worst': concave_points_worst, 'symmetry_worst': symmetry_worst}
    # print("*** Choices: " + choices)
    singleSampleDf = singleSampleDf.append(choices, ignore_index=True)
    singleSampleDf = sm.add_constant(singleSampleDf, has_constant='add')
    print(singleSampleDf)
    singlePrediction = loadedModel.predict(singleSampleDf)
    
    translated_prediction = int(singlePrediction[0])
    str_answer = ""
    if translated_prediction == 1:
        str_answer = "positive for Breast Cancer"
    else:
        str_answer = "negative for Breast Cancer"
    print("Single prediction: " + str_answer)
    return render(request, 'results.html', {'choices': choices, 'preds': str_answer})
