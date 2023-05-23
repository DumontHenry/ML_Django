from django.shortcuts import render
import pandas as pd
from django.http import JsonResponse
from .models import PredResults

# Create your views here.

def predict(request):
    return render(request,'predict.html')

def predict_chances(request):
    if request.POST.get('action')== 'post':
        sepal_length = float(request.POST.get('sepal_length'))
        sepal_width = float(request.POST.get('sepal_width'))
        petal_length = float(request.POST.get('petal_length'))
        petal_width = float(request.POST.get('petal_width'))

        model = pd.read_pickle(r"ML_Iris_web/ml_model/new_models.pkl")
        result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

        classification = result[0]

        PredResults.objects.create(sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length,
                                   petal_width=petal_width, classification=classification)

        return JsonResponse({'result': classification, 'sepal_length': sepal_length,
                             'sepal_width': sepal_width, 'petal_length': petal_length, 'petal_width': petal_width},
                            safe=False)

def view_results(request):
    # Submit prediction and show all
    data = {"dataset": PredResults.objects.all()}
    return render(request, "results.html", data)