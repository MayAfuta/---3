# 🎬 Movie Rating Predictor

אפליקציית Flask לחיזוי הדירוג הממוצע (averageRating) של סרט בטרם יציאתו לאקרנים,
המבוססת על מודל Random Forest שאומן בחלק 2 של המטלה.

## תיאור הפרויקט

האפליקציה מקבלת פרטי סרט (שנה, אורך, ז'אנר, שפה, מדינה, וקיום תקציב)
ומחזירה תחזית לדירוג הסרט בסקאלה של 1.0–10.0.
המערכת בנויה משלוש שכבות: ממשק HTML, שרת Flask, ומודל מאומן.

## מבנה הקבצים

| קובץ | תיאור |
|------|-------|
| index.html | דף HTML עם טופס הקלט וכפתור Predict |
| api.py | שרת Flask עם נקודות הקצה |
| assets_data_prep.py | פונקציית data_prepare() מחלק 2 |
| trained_model.pkl | המודל המאומן |
| requirements.txt | רשימת הספריות הנדרשות |
| README.md | קובץ זה |

## הוראות התקנה

1. שכפול ה-repository:

git clone <repository-url>
cd <repository-name>

2. יצירת סביבה וירטואלית:

python -m venv venv

3. הפעלת הסביבה הוירטואלית:

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

4. התקנת הספריות:

pip install -r requirements.txt

## הוראות הפעלה

הרצת השרת:

python api.py

כתובת גישה לאפליקציה:

http://localhost:5000

## שדות הקלט וטווחי הערכים

| שדה | סוג | טווח / ערכים |
|------|-----|---------------|
| Release Year | מספר | 1880–2030 |
| Runtime (minutes) | מספר | 1–500 |
| Genre | בחירה | Drama, Comedy, Romance, Action, Documentary, Crime, Thriller, Horror, Adventure, Mystery, Family, Fantasy, Biography, History, Sci-Fi, Music |
| Language | בחירה | English, French, Hindi, Spanish, Other/Unknown |
| Country | בחירה | United States, India, United Kingdom, France, Italy, Japan, Canada, Other/Unknown |
| Known Budget | בחירה | Yes / No |

## חברי הצוות

- יהודית אלמיהו — 206131468
- מאי אפוטה  — 315098707