# دليل نشر نظام قراءة الملكية على Vercel (UAE Mulkiya Reader)

تم بناء وتجهيز هذا المشروع ليعمل بشكل أصلي (Native) وخفيف وسريع جداً على **Vercel Serverless** مع الربط التلقائي بـ **Google Cloud Document AI** لحفظ الخصوصية التامة.

---

## 🚀 الخطوة 1: تشغيل المشروع محلياً واختباره
المشروع معد ومربوط مسبقاً بمفاتيحك في ملف `.env.local`:
```bash
# تشغيل خادم التطوير
npm run dev
```
افتح المتصفح على: **http://localhost:3000** (أو البورت الظاهر لديك).
يمكنك رفع أي صورة ملكية أو النقر على "Range Rover (Dubai)" لتجربة عينة مباشرة.

---

## 🌐 الخطوة 2: رفع الكود إلى GitHub
قم بإنشاء مستودع جديد على حسابك في GitHub، ثم نفّذ الأوامر التالية:

```bash
git init
git add .
git commit -m "feat: UAE Mulkiya Reader with Google Cloud Document AI"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/uae-mulkiya-reader.git
git push -u origin main
```

---

## ⚡ الخطوة 3: النشر على Vercel بضغطة زر
1. ادخل إلى حسابك في [Vercel](https://vercel.com) واضغط **Add New...** ← **Project**.
2. اختر مستودع الـ GitHub (`uae-mulkiya-reader`).
3. في قسم **Environment Variables** (المتغيرات البيئية)، أضف المتغيرات التالية (موجودة وجاهزة لديك في `.env.example`):

| اسم المتغير (Variable Name) | القيمة (Value) |
|---|---|
| `GCP_PROJECT_ID` | `aireader-507611` |
| `GCP_PROJECT_NUMBER` | `976524610604` |
| `GCP_LOCATION` | `eu` |
| `GCP_PROCESSOR_ID` | `33d1dea3952e2d2c` |
| `GCP_CLIENT_EMAIL` | `vercel@aireader-507611.iam.gserviceaccount.com` |
| `GCP_PRIVATE_KEY` | *(الصق المفتاح الخاص بالكامل كما هو مع الأسطر الجديدة)* |

4. اضغط **Deploy**!
خلال أقل من دقيقة، سيكون لديك رابط رسمي وسريع ومحمي بـ HTTPS ترسله للشركة لتجربته فوراً.

---

## 🔒 معايير الخصوصية والأمان المطبقة:
* **Zero-Disk Retention:** الصور تُعالج في الذاكرة المؤقتة (RAM) لـ Vercel وتُرسل مباشرة إلى Google Cloud المشفر في أوروبا (`eu`) ولا يتم حفظها على القرص.
* **No AI Training:** معالجة الوثائق تتم تحت مظلة اتفاقية Google Cloud Enterprise ولا تُستخدم البيانات لتدريب أي نماذج.
