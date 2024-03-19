from django.shortcuts import render
from .models import OneDomainPredictor

def home(request):
    if request.method == 'POST':
        domain = request.POST.get('domain')
        predictor = OneDomainPredictor(domain_name=domain)
        class_label = predictor.predict_one(domain)
        predictor.predicted_class = class_label
        predictor.save()
        return render(request, 'home.html', {'domain': domain,
                                             'class_label': class_label})
    else:
        return render(request, 'home.html')

