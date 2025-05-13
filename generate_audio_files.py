from gtts import gTTS
import os

# Arabic gesture words
words = [
    'الزائدة الدودية', 'العمود الفقري', 'الصدر', 'جهاز التنفس', 'الهيكل العظمي',
    'القصبة الهوائية', 'الوخز بالإبر', 'ضغط الدم', 'كبسولة', 'زكام', 'الجهاز الهضمي',
    'يشرب', 'قطارة', 'أدوية', 'صحي', 'يسمع', 'القلب', 'المناعة', 'يستنشق', 'تلقيح',
    'الكبد', 'دواء', 'ميكروب', 'منغولي', 'عضلة', 'البنكرياس', 'صيدلية', 'البلعوم',
    'إعاقة جسدية', 'فحص جسدي', 'تلقيح النباتات', 'نبض', 'فحص البصر', 'صمت',
    'جمجمة', 'نوم', 'سماعة الطبيب', 'فيروس', 'ضعف بصري', 'استيقاظ', 'جرح'
]

# Create the videos/ folder if not exists
os.makedirs("videos", exist_ok=True)

# Generate audio files
for word in words:
    filename = f"videos/{word}.mp3"
    if not os.path.exists(filename):
        try:
            tts = gTTS(text=word, lang='ar')
            tts.save(filename)
            print(f"✅ Saved: {filename}")
        except Exception as e:
            print(f"❌ Failed to save {word}: {e}")
    else:
        print(f"🟡 Already exists: {filename}")
