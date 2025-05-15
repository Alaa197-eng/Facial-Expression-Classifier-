# مشروع تصنيف تعابير الوجه باستخدام التعلم العميق

هذا المشروع يعتمد على نموذج ResNet18 مدرب باستخدام PyTorch لتصنيف تعابير الوجه (سعيد، غاضب، حزين، مفاجئ...).

## المميزات
- نقل تعلم لتقليل زمن التدريب
- واجهة Flask لرفع الصور والتصنيف
- التدريب باستخدام Jupyter Notebook

## التشغيل
1. ثبت المتطلبات:
```bash
pip install -r requirements.txt
```

2. درّب النموذج عبر `notebooks/training_notebook.ipynb`
3. شغّل الواجهة:
```bash
cd app
python app.py
```

## ملاحظة حول البيانات والملفات الكبيرة

لضمان خفة المشروع، لم يتم تضمين:
- مجلد `data/fer2013/`
- ملف `fer_model.pth`
- مجلد `uploads/`

يرجى تحميل مجموعة بيانات FER2013 يدويًا ووضعها في:
```
data/fer2013/train/
data/fer2013/test/
```
عبر الرابط:
https://www.kaggle.com/datasets/msambare/fer2013

ويمكنك توليد النموذج باستخدام الدفتر.

Facial_Expression_Classifier.ipynb