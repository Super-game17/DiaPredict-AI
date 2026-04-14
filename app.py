import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
from sklearn.impute import SimpleImputer
import qrcode
from io import BytesIO
import socket

# =============================================================================
# 1. CONFIGURATION PAGE (must be first)
# =============================================================================
st.set_page_config(
    page_title="DiaPredict Pro AI",
    # page_icon="✨",  # fallback, mais sera remplacé par le logo HTML
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# 2. INITIALISATION DE LA SESSION (langue & seuil)
# =============================================================================
if "language" not in st.session_state:
    st.session_state.language = "fr"  # fr, en, es
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.40  # seuil par défaut 40%

# =============================================================================
# 3. DICTIONNAIRES DE TRADUCTION (ajout des clés QR)
# =============================================================================
translations = {
    "fr": {
        "title": "DiaPredict Pro AI",
        "subtitle": "Prédiction intelligente du risque de diabète",
        "how_it_works": "🔬 Comment ça marche ?",
        "how_text": "Notre modèle d'intelligence artificielle analyse 8 paramètres cliniques clés et vous fournit une probabilité de risque, basée sur des données médicales réelles (Pima Indians).",
        "accuracy": "⚡ Précision : 80% (AUC)",
        "threshold_custom": "📊 Seuil personnalisé : ajusté pour minimiser les faux négatifs.",
        "anon": "🔒 Données anonymes",
        "ai_cert": "🤖 IA certifiée",
        "realtime": "Analyse en temps réel",
        "clinical_params": "📋 Paramètres cliniques",
        "pregnancies": "🤰 Grossesses",
        "glucose": "🩸 Glucose (mg/dL)",
        "blood_pressure": "❤️ Tension artérielle (mmHg)",
        "skin_thickness": "📏 Épaisseur peau (mm)",
        "insulin": "💉 Insuline (mu U/ml)",
        "bmi": "⚖️ IMC (kg/m²)",
        "dpf": "🧬 Fonction pedigree diabète",
        "age": "🎂 Âge (ans)",
        "predict_button": "🚀 Lancer l'analyse prédictive",
        "analyzing": "🧠 L'IA analyse vos données...",
        "threshold_label": "Seuil d'alerte",
        "risk": "Risque",
        "alert_high": "⚠️ Alerte : Risque Élevé Détecté",
        "alert_high_msg": "Votre probabilité dépasse le seuil clinique. Nous vous recommandons vivement de consulter un médecin généraliste ou un endocrinologue pour un bilan complet (HbA1c, glycémie à jeun).",
        "alert_high_advice": "✨ Des mesures précoces peuvent inverser la tendance.",
        "alert_low": "✅ Risque Faible – Bonnes nouvelles",
        "alert_low_msg": "Vos indicateurs sont dans une zone de confort. Maintenez une activité physique régulière et une alimentation équilibrée pour rester en bonne santé.",
        "alert_low_advice": "🔍 Réévaluez dans 6 mois ou plus tôt si des symptômes apparaissent.",
        "disclaimer": "⚠️ Cette application est un outil d'aide à la décision. Seul un professionnel de santé peut poser un diagnostic.",
        "footer": "DiaPredict AI – Propulsé par Streamlit & Scikit-learn | Données PIMA Indians",
        "language_select": "🌐 Langue",
        "threshold_slider": "🎚️ Seuil de risque critique",
        "qr_title": "📲 Accès rapide (QR code)",
        "qr_info": "Scannez ce code depuis un autre appareil sur le même réseau local.",
        "qr_url_label": "URL personnalisée (optionnel)",
    },
    "en": {
        "title": "DiaPredict Pro AI",
        "subtitle": "Intelligent diabetes risk prediction",
        "how_it_works": "🔬 How it works?",
        "how_text": "Our AI model analyzes 8 key clinical parameters and provides a risk probability based on real medical data (Pima Indians).",
        "accuracy": "⚡ Accuracy: 80% (AUC)",
        "threshold_custom": "📊 Custom threshold: adjusted to minimize false negatives.",
        "anon": "🔒 Anonymous data",
        "ai_cert": "🤖 AI certified",
        "realtime": "Real-time analysis",
        "clinical_params": "📋 Clinical parameters",
        "pregnancies": "🤰 Pregnancies",
        "glucose": "🩸 Glucose (mg/dL)",
        "blood_pressure": "❤️ Blood pressure (mmHg)",
        "skin_thickness": "📏 Skin thickness (mm)",
        "insulin": "💉 Insulin (mu U/ml)",
        "bmi": "⚖️ BMI (kg/m²)",
        "dpf": "🧬 Diabetes pedigree function",
        "age": "🎂 Age (years)",
        "predict_button": "🚀 Run predictive analysis",
        "analyzing": "🧠 AI is analyzing your data...",
        "threshold_label": "Alert threshold",
        "risk": "Risk",
        "alert_high": "⚠️ Alert: High Risk Detected",
        "alert_high_msg": "Your probability exceeds the clinical threshold. We strongly recommend consulting a general practitioner or endocrinologist for a complete check-up (HbA1c, fasting glucose).",
        "alert_high_advice": "✨ Early measures can reverse the trend.",
        "alert_low": "✅ Low Risk – Good news",
        "alert_low_msg": "Your indicators are in a comfortable zone. Maintain regular physical activity and a balanced diet to stay healthy.",
        "alert_low_advice": "🔍 Reassess in 6 months or earlier if symptoms appear.",
        "disclaimer": "⚠️ This application is a decision support tool. Only a healthcare professional can make a diagnosis.",
        "footer": "DiaPredict AI – Powered by Streamlit & Scikit-learn | PIMA Indians Dataset",
        "language_select": "🌐 Language",
        "threshold_slider": "🎚️ Critical risk threshold",
        "qr_title": "📲 Quick access (QR code)",
        "qr_info": "Scan this code from another device on the same local network.",
        "qr_url_label": "Custom URL (optional)",
    },
    "es": {
        "title": "DiaPredict Pro AI",
        "subtitle": "Predicción inteligente del riesgo de diabetes",
        "how_it_works": "🔬 ¿Cómo funciona?",
        "how_text": "Nuestro modelo de IA analiza 8 parámetros clínicos clave y proporciona una probabilidad de riesgo basada en datos médicos reales (Pima Indians).",
        "accuracy": "⚡ Precisión: 80% (AUC)",
        "threshold_custom": "📊 Umbral personalizado: ajustado para minimizar falsos negativos.",
        "anon": "🔒 Datos anónimos",
        "ai_cert": "🤖 IA certificada",
        "realtime": "Análisis en tiempo real",
        "clinical_params": "📋 Parámetros clínicos",
        "pregnancies": "🤰 Embarazos",
        "glucose": "🩸 Glucosa (mg/dL)",
        "blood_pressure": "❤️ Presión arterial (mmHg)",
        "skin_thickness": "📏 Grosor de la piel (mm)",
        "insulin": "💉 Insulina (mu U/ml)",
        "bmi": "⚖️ IMC (kg/m²)",
        "dpf": "🧬 Función de pedigrí diabético",
        "age": "🎂 Edad (años)",
        "predict_button": "🚀 Ejecutar análisis predictivo",
        "analyzing": "🧠 La IA está analizando tus datos...",
        "threshold_label": "Umbral de alerta",
        "risk": "Riesgo",
        "alert_high": "⚠️ Alerta: Riesgo Alto Detectado",
        "alert_high_msg": "Tu probabilidad supera el umbral clínico. Recomendamos encarecidamente consultar a un médico de cabecera o endocrinólogo para un chequeo completo (HbA1c, glucosa en ayunas).",
        "alert_high_advice": "✨ Las medidas tempranas pueden revertir la tendencia.",
        "alert_low": "✅ Bajo Riesgo – Buenas noticias",
        "alert_low_msg": "Tus indicadores están en una zona cómoda. Mantén actividad física regular y una dieta equilibrada para mantenerte saludable.",
        "alert_low_advice": "🔍 Reevalúa en 6 meses o antes si aparecen síntomas.",
        "disclaimer": "⚠️ Esta aplicación es una herramienta de apoyo a la decisión. Solo un profesional de la salud puede hacer un diagnóstico.",
        "footer": "DiaPredict AI – Desarrollado con Streamlit & Scikit-learn | Datos PIMA Indians",
        "language_select": "🌐 Idioma",
        "threshold_slider": "🎚️ Umbral de riesgo crítico",
        "qr_title": "📲 Acceso rápido (código QR)",
        "qr_info": "Escanee este código desde otro dispositivo en la misma red local.",
        "qr_url_label": "URL personalizada (opcional)",
    }
}

def t(key):
    """Helper de traduction."""
    return translations[st.session_state.language].get(key, key)


# =============================================================================
# 3b. CLINICAL INDICATORS (all model input factors)
# =============================================================================
def analyze_key_indicators(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age, language):
    """Return patient-friendly indicator interpretation and advice."""
    i18n = {
        "fr": {
            "title": "Indicateurs cliniques personnalisés",
            "subtitle": "Pourquoi le risque peut monter ou rester modéré selon vos entrées.",
            "up_title": "Facteurs qui peuvent augmenter le risque",
            "down_title": "Facteurs plutôt rassurants",
            "none_up": "Aucun signal majeur sur ces 3 paramètres.",
            "none_down": "Peu d'éléments protecteurs sur ces 3 paramètres.",
            "status": {
                "low": "Bas",
                "normal": "Zone cible",
                "elevated": "À surveiller",
                "high": "Élevé",
            },
            "glucose": {
                "label": "Glucose",
                "what": "Le glucose reflète le taux de sucre dans le sang.",
                "unit": "mg/dL",
            },
            "bp": {
                "label": "Pression artérielle (diastolique)",
                "what": "La pression artérielle mesure la force du sang dans les artères.",
                "unit": "mmHg",
            },
            "insulin": {
                "label": "Insuline",
                "what": "L'insuline aide le glucose à entrer dans les cellules.",
                "unit": "mu U/ml",
            },
            "pregnancies": {
                "label": "Grossesses",
                "what": "Le nombre de grossesses peut influencer le risque métabolique selon le contexte global.",
                "unit": "",
            },
            "skin": {
                "label": "Epaisseur de peau",
                "what": "L'epaisseur du pli cutane est un indicateur indirect de composition corporelle.",
                "unit": "mm",
            },
            "bmi": {
                "label": "IMC",
                "what": "L'IMC estime la corpulence et s'associe au risque de resistance a l'insuline.",
                "unit": "kg/m2",
            },
            "dpf": {
                "label": "Pedigree diabete",
                "what": "Le score pedigree reflete une predisposition familiale au diabete.",
                "unit": "",
            },
            "age": {
                "label": "Age",
                "what": "Le risque de diabete augmente generalement avec l'age.",
                "unit": "ans",
            },
        },
        "en": {
            "title": "Personalized clinical indicators",
            "subtitle": "Why risk can increase or stay moderate based on your entries.",
            "up_title": "Factors that may increase risk",
            "down_title": "Rather reassuring factors",
            "none_up": "No major warning signal on these 3 parameters.",
            "none_down": "Few protective elements on these 3 parameters.",
            "status": {
                "low": "Low",
                "normal": "Target range",
                "elevated": "Needs monitoring",
                "high": "High",
            },
            "glucose": {
                "label": "Glucose",
                "what": "Glucose reflects blood sugar level.",
                "unit": "mg/dL",
            },
            "bp": {
                "label": "Blood pressure (diastolic)",
                "what": "Blood pressure measures the force of blood in arteries.",
                "unit": "mmHg",
            },
            "insulin": {
                "label": "Insulin",
                "what": "Insulin helps glucose enter cells.",
                "unit": "mu U/ml",
            },
            "pregnancies": {
                "label": "Pregnancies",
                "what": "Number of pregnancies can influence metabolic risk depending on overall context.",
                "unit": "",
            },
            "skin": {
                "label": "Skin thickness",
                "what": "Skin fold thickness is an indirect indicator of body composition.",
                "unit": "mm",
            },
            "bmi": {
                "label": "BMI",
                "what": "BMI estimates body fatness and is linked to insulin resistance risk.",
                "unit": "kg/m2",
            },
            "dpf": {
                "label": "Diabetes pedigree",
                "what": "Pedigree score reflects family predisposition to diabetes.",
                "unit": "",
            },
            "age": {
                "label": "Age",
                "what": "Diabetes risk generally increases with age.",
                "unit": "years",
            },
        },
        "es": {
            "title": "Indicadores clínicos personalizados",
            "subtitle": "Por qué el riesgo puede subir o mantenerse moderado según tus datos.",
            "up_title": "Factores que pueden aumentar el riesgo",
            "down_title": "Factores más tranquilizadores",
            "none_up": "No hay señales mayores en estos 3 parámetros.",
            "none_down": "Pocos elementos protectores en estos 3 parámetros.",
            "status": {
                "low": "Bajo",
                "normal": "Rango objetivo",
                "elevated": "A vigilar",
                "high": "Alto",
            },
            "glucose": {
                "label": "Glucosa",
                "what": "La glucosa refleja el nivel de azúcar en sangre.",
                "unit": "mg/dL",
            },
            "bp": {
                "label": "Presión arterial (diastólica)",
                "what": "La presión arterial mide la fuerza de la sangre en las arterias.",
                "unit": "mmHg",
            },
            "insulin": {
                "label": "Insulina",
                "what": "La insulina ayuda a que la glucosa entre en las células.",
                "unit": "mu U/ml",
            },
            "pregnancies": {
                "label": "Embarazos",
                "what": "El numero de embarazos puede influir en el riesgo metabolico segun el contexto global.",
                "unit": "",
            },
            "skin": {
                "label": "Grosor de la piel",
                "what": "El grosor del pliegue cutaneo es un indicador indirecto de composicion corporal.",
                "unit": "mm",
            },
            "bmi": {
                "label": "IMC",
                "what": "El IMC estima la corpulencia y se asocia con riesgo de resistencia a la insulina.",
                "unit": "kg/m2",
            },
            "dpf": {
                "label": "Pedigrí diabetes",
                "what": "La puntuacion de pedigrí refleja predisposicion familiar a la diabetes.",
                "unit": "",
            },
            "age": {
                "label": "Edad",
                "what": "El riesgo de diabetes generalmente aumenta con la edad.",
                "unit": "anos",
            },
        },
    }

    lang = i18n.get(language, i18n["en"])
    normal_ranges = {
        "fr": {
            "pregnancies": "0-5",
            "glucose": "70-99 mg/dL (a jeun)",
            "bp": "60-79 mmHg (diastolique)",
            "skin": "10-35 mm",
            "insulin": "16-166 mu U/ml",
            "bmi": "18.5-24.9 kg/m2",
            "dpf": "0.20-0.80",
            "age": "21-34 ans",
        },
        "en": {
            "pregnancies": "0-5",
            "glucose": "70-99 mg/dL (fasting)",
            "bp": "60-79 mmHg (diastolic)",
            "skin": "10-35 mm",
            "insulin": "16-166 mu U/ml",
            "bmi": "18.5-24.9 kg/m2",
            "dpf": "0.20-0.80",
            "age": "21-34 years",
        },
        "es": {
            "pregnancies": "0-5",
            "glucose": "70-99 mg/dL (en ayunas)",
            "bp": "60-79 mmHg (diastolica)",
            "skin": "10-35 mm",
            "insulin": "16-166 mu U/ml",
            "bmi": "18.5-24.9 kg/m2",
            "dpf": "0.20-0.80",
            "age": "21-34 anos",
        },
    }
    ref = normal_ranges.get(language, normal_ranges["en"])

    texts = {
        "fr": {
            "why_label": "Pourquoi",
            "advice_label": "Conseil",
            "normal_word": "normal",
        },
        "en": {
            "why_label": "Why",
            "advice_label": "Advice",
            "normal_word": "normal",
        },
        "es": {
            "why_label": "Por que",
            "advice_label": "Consejo",
            "normal_word": "normal",
        },
    }
    ui_txt = texts.get(language, texts["en"])

    def classify_glucose(v):
        if v < 70:
            return "low"
        if v <= 99:
            return "normal"
        if v <= 125:
            return "elevated"
        return "high"

    def classify_bp(v):
        if v < 60:
            return "low"
        if v <= 79:
            return "normal"
        if v <= 89:
            return "elevated"
        return "high"

    def classify_insulin(v):
        if v < 16:
            return "low"
        if v <= 166:
            return "normal"
        return "high"

    def classify_pregnancies(v):
        if v <= 5:
            return "normal"
        if v <= 9:
            return "elevated"
        return "high"

    def classify_skin(v):
        if v < 10:
            return "low"
        if v <= 35:
            return "normal"
        if v <= 44:
            return "elevated"
        return "high"

    def classify_bmi(v):
        if v < 18.5:
            return "low"
        if v <= 24.9:
            return "normal"
        if v <= 29.9:
            return "elevated"
        return "high"

    def classify_dpf(v):
        if v < 0.2:
            return "low"
        if v <= 0.8:
            return "normal"
        if v <= 1.2:
            return "elevated"
        return "high"

    def classify_age(v):
        if v <= 34:
            return "normal"
        if v <= 49:
            return "elevated"
        return "high"

    statuses = {
        "pregnancies": classify_pregnancies(pregnancies),
        "glucose": classify_glucose(glucose),
        "bp": classify_bp(blood_pressure),
        "skin": classify_skin(skin_thickness),
        "insulin": classify_insulin(insulin),
        "bmi": classify_bmi(bmi),
        "dpf": classify_dpf(dpf),
        "age": classify_age(age),
    }

    details = []
    up_factors = []
    down_factors = []

    ordered_values = [
        ("pregnancies", pregnancies),
        ("glucose", glucose),
        ("bp", blood_pressure),
        ("skin", skin_thickness),
        ("insulin", insulin),
        ("bmi", bmi),
        ("dpf", dpf),
        ("age", age),
    ]

    for key, value in ordered_values:
        status = statuses[key]
        base = lang[key]
        status_text = lang["status"][status]

        if key == "glucose":
            if status in ["elevated", "high"]:
                why = {
                    "fr": f"Un glucose élevé peut traduire une difficulté à réguler le sucre (résistance à l'insuline). La zone normale est {ref['glucose']}.",
                    "en": f"High glucose may reflect reduced sugar regulation (insulin resistance). The normal range is {ref['glucose']}.",
                    "es": f"La glucosa alta puede reflejar una regulación deficiente del azúcar (resistencia a la insulina). El rango normal es {ref['glucose']}.",
                }[language]
                advice = {
                    "fr": "Réduire les sucres rapides, augmenter les fibres, et contrôler la glycémie à jeun/HbA1c.",
                    "en": "Reduce fast sugars, increase fiber, and check fasting glucose/HbA1c.",
                    "es": "Reducir azúcares rápidos, aumentar fibra y controlar glucosa en ayunas/HbA1c.",
                }[language]
                up_factors.append(f"{base['label']}: {status_text} ({value:.0f} {base['unit']}) - normal: {ref['glucose']}")
            elif status == "normal":
                why = {
                    "fr": f"Un glucose dans la zone cible est un signal plutôt rassurant. La zone normale est {ref['glucose']}.",
                    "en": f"Glucose in target range is a reassuring signal. The normal range is {ref['glucose']}.",
                    "es": f"La glucosa en rango objetivo es una señal tranquilizadora. El rango normal es {ref['glucose']}.",
                }[language]
                advice = {
                    "fr": "Maintenir une alimentation équilibrée et une activité physique régulière.",
                    "en": "Maintain balanced nutrition and regular physical activity.",
                    "es": "Mantener alimentación equilibrada y actividad física regular.",
                }[language]
                down_factors.append(f"{base['label']}: {status_text} ({value:.0f} {base['unit']}) - normal: {ref['glucose']}")
            else:
                why = {
                    "fr": f"Un glucose très bas n'indique pas un diabète à lui seul mais reste à surveiller cliniquement. La zone normale est {ref['glucose']}.",
                    "en": f"Very low glucose alone does not indicate diabetes but still deserves clinical review. The normal range is {ref['glucose']}.",
                    "es": f"La glucosa muy baja por sí sola no indica diabetes, pero requiere revisión clínica. El rango normal es {ref['glucose']}.",
                }[language]
                advice = {
                    "fr": "Ne pas sauter les repas et discuter des symptômes avec un professionnel de santé.",
                    "en": "Avoid skipping meals and discuss symptoms with a healthcare professional.",
                    "es": "No saltar comidas y comentar síntomas con un profesional de salud.",
                }[language]
        elif key == "bp":
            if status in ["elevated", "high"]:
                why = {
                    "fr": f"Une pression artérielle élevée est souvent associée au syndrome métabolique et au risque cardiométabolique. La zone normale est {ref['bp']}.",
                    "en": f"Elevated blood pressure is often associated with metabolic syndrome and cardiometabolic risk. The normal range is {ref['bp']}.",
                    "es": f"La presión arterial elevada suele asociarse al síndrome metabólico y riesgo cardiometabólico. El rango normal es {ref['bp']}.",
                }[language]
                advice = {
                    "fr": "Limiter le sel, surveiller le poids, et contrôler la tension régulièrement.",
                    "en": "Reduce salt, monitor weight, and check blood pressure regularly.",
                    "es": "Limitar la sal, vigilar el peso y controlar la presión regularmente.",
                }[language]
                up_factors.append(f"{base['label']}: {status_text} ({value:.0f} {base['unit']}) - normal: {ref['bp']}")
            elif status == "normal":
                why = {
                    "fr": f"Une pression artérielle en zone cible est favorable au profil global de risque. La zone normale est {ref['bp']}.",
                    "en": f"Blood pressure in target range supports a healthier risk profile. The normal range is {ref['bp']}.",
                    "es": f"Una presión arterial en rango objetivo favorece un mejor perfil de riesgo. El rango normal es {ref['bp']}.",
                }[language]
                advice = {
                    "fr": "Continuer l'activité physique et une alimentation pauvre en sel.",
                    "en": "Continue physical activity and a low-salt diet.",
                    "es": "Continuar actividad física y dieta baja en sal.",
                }[language]
                down_factors.append(f"{base['label']}: {status_text} ({value:.0f} {base['unit']}) - normal: {ref['bp']}")
            else:
                why = {
                    "fr": f"Une pression artérielle basse n'est pas un marqueur direct du diabète, mais peut donner fatigue ou vertiges. La zone normale est {ref['bp']}.",
                    "en": f"Low blood pressure is not a direct diabetes marker, but can cause fatigue or dizziness. The normal range is {ref['bp']}.",
                    "es": f"La presión baja no es un marcador directo de diabetes, pero puede causar fatiga o mareo. El rango normal es {ref['bp']}.",
                }[language]
                advice = {
                    "fr": "S'hydrater et vérifier en cas de symptômes persistants.",
                    "en": "Stay hydrated and reassess if symptoms persist.",
                    "es": "Hidratarse y reevaluar si los síntomas persisten.",
                }[language]
        elif key == "pregnancies":
            if status in ["elevated", "high"]:
                why = {
                    "fr": f"Un nombre eleve de grossesses peut etre associe a un risque metabolique plus eleve selon le contexte. La zone normale est {ref['pregnancies']}.",
                    "en": f"A higher number of pregnancies can be associated with increased metabolic risk depending on context. The normal range is {ref['pregnancies']}.",
                    "es": f"Un numero elevado de embarazos puede asociarse a mayor riesgo metabolico segun el contexto. El rango normal es {ref['pregnancies']}.",
                }[language]
                advice = {
                    "fr": "Surveiller la glycemie regulierement, surtout avec antecedents de diabete gestationnel.",
                    "en": "Monitor glucose regularly, especially with a history of gestational diabetes.",
                    "es": "Controlar la glucosa regularmente, sobre todo con antecedente de diabetes gestacional.",
                }[language]
                up_factors.append(f"{base['label']}: {status_text} ({value:.0f}) - {ui_txt['normal_word']}: {ref['pregnancies']}")
            else:
                why = {
                    "fr": f"Le nombre de grossesses se situe dans une zone courante. La zone normale est {ref['pregnancies']}.",
                    "en": f"The number of pregnancies is in a common range. The normal range is {ref['pregnancies']}.",
                    "es": f"El numero de embarazos esta en un rango habitual. El rango normal es {ref['pregnancies']}.",
                }[language]
                advice = {
                    "fr": "Maintenir un suivi medical preventif regulier.",
                    "en": "Maintain regular preventive medical follow-up.",
                    "es": "Mantener seguimiento medico preventivo regular.",
                }[language]
                down_factors.append(f"{base['label']}: {status_text} ({value:.0f}) - {ui_txt['normal_word']}: {ref['pregnancies']}")
        elif key == "skin":
            if status in ["elevated", "high"]:
                why = {
                    "fr": f"Une epaisseur elevee peut refleter une adiposite accrue, souvent associee a la resistance a l'insuline. La zone normale est {ref['skin']}.",
                    "en": f"Higher skin thickness may reflect increased adiposity, often associated with insulin resistance. The normal range is {ref['skin']}.",
                    "es": f"Un grosor elevado puede reflejar mayor adiposidad, asociada a resistencia a la insulina. El rango normal es {ref['skin']}.",
                }[language]
                advice = {
                    "fr": "Associer activite physique reguliere et alimentation equilibree.",
                    "en": "Combine regular physical activity with balanced nutrition.",
                    "es": "Combinar actividad fisica regular con alimentacion equilibrada.",
                }[language]
                up_factors.append(f"{base['label']}: {status_text} ({value:.0f} {base['unit']}) - {ui_txt['normal_word']}: {ref['skin']}")
            elif status == "normal":
                why = {
                    "fr": f"L'epaisseur de peau est dans une plage habituelle. La zone normale est {ref['skin']}.",
                    "en": f"Skin thickness is in a common range. The normal range is {ref['skin']}.",
                    "es": f"El grosor de piel esta en un rango habitual. El rango normal es {ref['skin']}.",
                }[language]
                advice = {
                    "fr": "Conserver les habitudes de vie favorables.",
                    "en": "Keep favorable lifestyle habits.",
                    "es": "Mantener habitos de vida favorables.",
                }[language]
                down_factors.append(f"{base['label']}: {status_text} ({value:.0f} {base['unit']}) - {ui_txt['normal_word']}: {ref['skin']}")
            else:
                why = {
                    "fr": f"Une valeur tres basse est peu informative seule et depend du contexte de mesure. La zone normale est {ref['skin']}.",
                    "en": f"A very low value alone is less informative and depends on measurement context. The normal range is {ref['skin']}.",
                    "es": f"Un valor muy bajo por si solo es poco informativo y depende del contexto de medicion. El rango normal es {ref['skin']}.",
                }[language]
                advice = {
                    "fr": "Verifier la mesure et interpreter avec l'IMC et le glucose.",
                    "en": "Recheck the measurement and interpret with BMI and glucose.",
                    "es": "Revisar la medicion e interpretar con IMC y glucosa.",
                }[language]
        elif key == "bmi":
            if status in ["elevated", "high"]:
                why = {
                    "fr": f"Un IMC eleve est un facteur majeur de resistance a l'insuline. La zone normale est {ref['bmi']}.",
                    "en": f"Higher BMI is a major insulin-resistance risk factor. The normal range is {ref['bmi']}.",
                    "es": f"Un IMC elevado es un factor importante de resistencia a la insulina. El rango normal es {ref['bmi']}.",
                }[language]
                advice = {
                    "fr": "Viser une reduction progressive du poids avec activite et nutrition adaptee.",
                    "en": "Aim for gradual weight reduction with activity and adapted nutrition.",
                    "es": "Buscar reduccion gradual de peso con actividad y nutricion adaptada.",
                }[language]
                up_factors.append(f"{base['label']}: {status_text} ({value:.1f} {base['unit']}) - {ui_txt['normal_word']}: {ref['bmi']}")
            elif status == "normal":
                why = {
                    "fr": f"Un IMC en zone cible est un facteur plutot protecteur. La zone normale est {ref['bmi']}.",
                    "en": f"BMI in target range is a rather protective factor. The normal range is {ref['bmi']}.",
                    "es": f"Un IMC en rango objetivo es un factor mas protector. El rango normal es {ref['bmi']}.",
                }[language]
                advice = {
                    "fr": "Maintenir les habitudes actuelles et la regularite sportive.",
                    "en": "Maintain current habits and regular exercise.",
                    "es": "Mantener habitos actuales y ejercicio regular.",
                }[language]
                down_factors.append(f"{base['label']}: {status_text} ({value:.1f} {base['unit']}) - {ui_txt['normal_word']}: {ref['bmi']}")
            else:
                why = {
                    "fr": f"Un IMC bas n'indique pas un diabete mais necessite un contexte nutritionnel. La zone normale est {ref['bmi']}.",
                    "en": f"Low BMI does not indicate diabetes but should be interpreted with nutritional context. The normal range is {ref['bmi']}.",
                    "es": f"Un IMC bajo no indica diabetes pero debe interpretarse con contexto nutricional. El rango normal es {ref['bmi']}.",
                }[language]
                advice = {
                    "fr": "Verifier l'equilibre nutritionnel avec un professionnel de sante.",
                    "en": "Check nutritional balance with a healthcare professional.",
                    "es": "Revisar equilibrio nutricional con un profesional de salud.",
                }[language]
        elif key == "dpf":
            if status in ["elevated", "high"]:
                why = {
                    "fr": f"Un score pedigree eleve suggere une predisposition familiale plus importante. La zone normale est {ref['dpf']}.",
                    "en": f"A higher pedigree score suggests stronger family predisposition. The normal range is {ref['dpf']}.",
                    "es": f"Una puntuacion de pedigrí elevada sugiere mayor predisposicion familiar. El rango normal es {ref['dpf']}.",
                }[language]
                advice = {
                    "fr": "Renforcer la prevention: depistage regulier et hygiene de vie stricte.",
                    "en": "Strengthen prevention: regular screening and strict lifestyle habits.",
                    "es": "Reforzar prevencion: control regular y habitos de vida estrictos.",
                }[language]
                up_factors.append(f"{base['label']}: {status_text} ({value:.3f}) - {ui_txt['normal_word']}: {ref['dpf']}")
            elif status == "normal":
                why = {
                    "fr": f"Le score pedigree est dans une zone intermediaire habituelle. La zone normale est {ref['dpf']}.",
                    "en": f"Pedigree score is in a common intermediate range. The normal range is {ref['dpf']}.",
                    "es": f"La puntuacion de pedigrí esta en un rango intermedio habitual. El rango normal es {ref['dpf']}.",
                }[language]
                advice = {
                    "fr": "Poursuivre une prevention reguliere.",
                    "en": "Continue regular prevention.",
                    "es": "Continuar prevencion regular.",
                }[language]
                down_factors.append(f"{base['label']}: {status_text} ({value:.3f}) - {ui_txt['normal_word']}: {ref['dpf']}")
            else:
                why = {
                    "fr": f"Un score faible est plutot rassurant sur l'heritabilite. La zone normale est {ref['dpf']}.",
                    "en": f"A low score is rather reassuring regarding inherited risk. The normal range is {ref['dpf']}.",
                    "es": f"Una puntuacion baja es mas tranquilizadora sobre riesgo hereditario. El rango normal es {ref['dpf']}.",
                }[language]
                advice = {
                    "fr": "Conserver un mode de vie protecteur.",
                    "en": "Keep a protective lifestyle.",
                    "es": "Mantener un estilo de vida protector.",
                }[language]
                down_factors.append(f"{base['label']}: {status_text} ({value:.3f}) - {ui_txt['normal_word']}: {ref['dpf']}")
        elif key == "age":
            if status in ["elevated", "high"]:
                why = {
                    "fr": f"Avec l'age, le risque de diabete de type 2 tend a augmenter. La zone normale est {ref['age']}.",
                    "en": f"With age, type 2 diabetes risk tends to increase. The normal range is {ref['age']}.",
                    "es": f"Con la edad, el riesgo de diabetes tipo 2 tiende a aumentar. El rango normal es {ref['age']}.",
                }[language]
                advice = {
                    "fr": "Faire un depistage metabolique regulier (glycemie, HbA1c, bilan medical).",
                    "en": "Perform regular metabolic screening (glucose, HbA1c, medical check-up).",
                    "es": "Realizar control metabolico regular (glucosa, HbA1c, revision medica).",
                }[language]
                up_factors.append(f"{base['label']}: {status_text} ({value:.0f} {base['unit']}) - {ui_txt['normal_word']}: {ref['age']}")
            else:
                why = {
                    "fr": f"Un age plus jeune est generalement associe a un risque de base plus faible. La zone normale est {ref['age']}.",
                    "en": f"Younger age is generally associated with lower baseline risk. The normal range is {ref['age']}.",
                    "es": f"Una edad mas joven suele asociarse con menor riesgo basal. El rango normal es {ref['age']}.",
                }[language]
                advice = {
                    "fr": "Maintenir les habitudes preventives pour conserver cet avantage.",
                    "en": "Maintain preventive habits to preserve this advantage.",
                    "es": "Mantener habitos preventivos para conservar esta ventaja.",
                }[language]
                down_factors.append(f"{base['label']}: {status_text} ({value:.0f} {base['unit']}) - {ui_txt['normal_word']}: {ref['age']}")
        else:
            if status == "high":
                why = {
                    "fr": f"Une insuline élevée peut suggérer une résistance à l'insuline, fréquemment liée au prédiabète. La zone normale est {ref['insulin']}.",
                    "en": f"High insulin may suggest insulin resistance, often linked to prediabetes. The normal range is {ref['insulin']}.",
                    "es": f"La insulina alta puede sugerir resistencia a la insulina, a menudo ligada a prediabetes. El rango normal es {ref['insulin']}.",
                }[language]
                advice = {
                    "fr": "Viser une perte de poids progressive, activité d'endurance et suivi médical.",
                    "en": "Target gradual weight loss, endurance activity, and medical follow-up.",
                    "es": "Buscar pérdida de peso gradual, actividad de resistencia y seguimiento médico.",
                }[language]
                up_factors.append(f"{base['label']}: {status_text} ({value:.0f} {base['unit']}) - normal: {ref['insulin']}")
            elif status == "normal":
                why = {
                    "fr": f"Une insuline en zone habituelle est plutôt cohérente avec une bonne réponse métabolique. La zone normale est {ref['insulin']}.",
                    "en": f"Insulin in a typical range is generally consistent with a healthier metabolic response. The normal range is {ref['insulin']}.",
                    "es": f"Una insulina en rango habitual suele ser coherente con una mejor respuesta metabólica. El rango normal es {ref['insulin']}.",
                }[language]
                advice = {
                    "fr": "Conserver des repas équilibrés et éviter la sédentarité prolongée.",
                    "en": "Keep balanced meals and avoid prolonged sedentary behavior.",
                    "es": "Mantener comidas equilibradas y evitar el sedentarismo prolongado.",
                }[language]
                down_factors.append(f"{base['label']}: {status_text} ({value:.0f} {base['unit']}) - normal: {ref['insulin']}")
            else:
                why = {
                    "fr": f"Une insuline basse isolée n'explique pas à elle seule le risque de diabète. La zone normale est {ref['insulin']}.",
                    "en": f"Low insulin alone does not fully explain diabetes risk. The normal range is {ref['insulin']}.",
                    "es": f"La insulina baja por sí sola no explica totalmente el riesgo de diabetes. El rango normal es {ref['insulin']}.",
                }[language]
                advice = {
                    "fr": "Interpréter ce résultat avec le glucose et l'avis d'un professionnel.",
                    "en": "Interpret this result together with glucose and clinical advice.",
                    "es": "Interpretar este resultado junto con la glucosa y consejo clínico.",
                }[language]

        value_formats = {
            "pregnancies": f"{value:.0f}",
            "glucose": f"{value:.0f}",
            "bp": f"{value:.0f}",
            "skin": f"{value:.0f}",
            "insulin": f"{value:.1f}",
            "bmi": f"{value:.1f}",
            "dpf": f"{value:.3f}",
            "age": f"{value:.0f}",
        }

        details.append({
            "label": base["label"],
            "what": base["what"],
            "value": value_formats.get(key, f"{value:.0f}"),
            "unit": base["unit"],
            "status": status_text,
            "status_key": status,
            "why": why,
            "advice": advice,
            "why_label": ui_txt["why_label"],
            "advice_label": ui_txt["advice_label"],
        })

    return {
        "title": lang["title"],
        "subtitle": lang["subtitle"],
        "up_title": lang["up_title"],
        "down_title": lang["down_title"],
        "none_up": lang["none_up"],
        "none_down": lang["none_down"],
        "details": details,
        "up_factors": up_factors,
        "down_factors": down_factors,
    }

# =============================================================================
# 4. CUSTOM CSS (WOW EFFECT + LOGO STYLE) – INCHANGÉ
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,500;14..32,700;14..32,800&display=swap');

/* Global reset & background animated gradient */
.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(2, 10, 30, 1) 0%, rgba(10, 5, 25, 1) 100%);
    font-family: 'Inter', sans-serif;
    overflow-x: hidden;
}

/* Animated floating particles effect */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: radial-gradient(circle at 20% 40%, rgba(56, 189, 248, 0.08) 1px, transparent 1px),
                      radial-gradient(circle at 80% 70%, rgba(139, 92, 246, 0.08) 1px, transparent 1px);
    background-size: 50px 50px, 80px 80px;
    pointer-events: none;
    z-index: 0;
    animation: floatParticles 20s linear infinite;
}

@keyframes floatParticles {
    0% { background-position: 0 0, 0 0; }
    100% { background-position: 100px 100px, 150px 80px; }
}

/* Glass morphism cards */
.glass-card {
    background: rgba(15, 25, 45, 0.55);
    backdrop-filter: blur(12px);
    border-radius: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
    padding: 1.8rem;
    transition: all 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-5px);
    border-color: rgba(56, 189, 248, 0.5);
    box-shadow: 0 25px 40px rgba(0, 0, 0, 0.4);
}

/* Animated gradient text */
.gradient-text {
    font-size: 3rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #38BDF8, #A78BFA, #F472B6);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}

/* Modern input fields */
.stNumberInput > div > div > input {
    background: rgba(10, 20, 40, 0.7) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 1rem !important;
    color: white !important;
    font-weight: 500 !important;
    padding: 0.7rem 1rem !important;
    transition: all 0.2s;
}

.stNumberInput > div > div > input:focus {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
}

/* Labels */
label p {
    font-weight: 600 !important;
    color: #CBD5E1 !important;
    letter-spacing: 0.3px;
}

/* Glowing button */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2563EB, #7C3AED) !important;
    border: none !important;
    padding: 0.9rem !important;
    border-radius: 1.5rem !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    color: white !important;
    transition: all 0.3s ease;
    box-shadow: 0 5px 15px rgba(37, 99, 235, 0.3);
}

.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.5);
    background: linear-gradient(90deg, #3B82F6, #8B5CF6) !important;
}

/* Custom logo style */
.custom-logo {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 1rem;
}
.logo-circle {
    width: 70px;
    height: 70px;
    background: rgba(168, 85, 247, 0.15);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 2px solid #A78BFA;
    box-shadow: 0 0 15px rgba(168, 85, 247, 0.3);
}
.logo-circle span {
    font-size: 2rem;
    font-weight: bold;
    background: linear-gradient(135deg, #38BDF8, #F472B6);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.logo-text {
    margin-left: 1rem;
    font-size: 1.8rem;
    font-weight: 800;
}
.logo-text .dia {
    background: linear-gradient(135deg, #38BDF8, #A78BFA);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.logo-text .ai {
    background: linear-gradient(135deg, #F472B6, #A78BFA);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

/* Result cards */
.result-card {
    border-radius: 1.5rem;
    padding: 1.2rem;
    margin-top: 1rem;
    backdrop-filter: blur(8px);
    animation: fadeSlideUp 0.5s ease-out;
}

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Custom divider */
.custom-hr {
    margin: 1.5rem 0;
    border: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #38BDF8, #A78BFA, transparent);
}

/* Sidebar styling */
.css-1d391kg, .css-12oz5g7 {
    background: rgba(10, 20, 40, 0.7) !important;
    backdrop-filter: blur(10px);
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 5. LOAD MODEL & IMPUTER (with fallback)
# =============================================================================
@st.cache_resource
def load_assets():
    try:
        artifacts = joblib.load('modele_ml.pkl')
        model = artifacts.get('model')
        threshold = artifacts.get('threshold', 0.5)
        return model, threshold, None
    except:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        dummy_model = make_pipeline(
            SimpleImputer(strategy='median'),
            LogisticRegression(random_state=42)
        )
        dummy_X = np.random.randn(100, 8)
        dummy_y = np.random.randint(0, 2, 100)
        dummy_model.fit(dummy_X, dummy_y)
        st.warning("⚠️ Modèle réel introuvable. Utilisation d'un modèle de démonstration. Veuillez placer 'modele_ml.pkl' pour des prédictions fiables.")
        return dummy_model, 0.5, None

model, _, external_imputer = load_assets()

# =============================================================================
# 6. HELPER: IMPUTATION
# =============================================================================
def preprocess_input(df):
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    if not hasattr(model, 'named_steps') and external_imputer is not None:
        medians = {"Glucose": 117.0, "BloodPressure": 72.0, "SkinThickness": 23.0, "Insulin": 30.0, "BMI": 32.0}
        for col in zero_cols:
            df[col] = df[col].fillna(medians[col])
    return df

# =============================================================================
# 7. FONCTION POUR OBTENIR L'IP LOCALE
# =============================================================================
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

# =============================================================================
# 8. SIDEBAR (LANGUAGE & THRESHOLD & QR CODE)
# =============================================================================
with st.sidebar:
    st.markdown("---")
    # Sélecteur de langue
    lang_map = {"Français": "fr", "English": "en", "Español": "es"}
    selected_lang_display = st.selectbox(
        t("language_select"),
        options=list(lang_map.keys()),
        index=list(lang_map.values()).index(st.session_state.language)
    )
    st.session_state.language = lang_map[selected_lang_display]
    
    # Curseur de seuil
    new_threshold = st.slider(
        t("threshold_slider"),
        min_value=0.0, max_value=1.0, value=st.session_state.threshold, step=0.01,
        help="Seuil de probabilité au-dessus duquel le risque est considéré comme élevé."
    )
    st.session_state.threshold = new_threshold
    
    st.markdown("---")
    
    # QR CODE (ajout)
    st.markdown(f"### {t('qr_title')}")
    st.caption(t("qr_info"))
    local_ip = get_local_ip()
    default_url = f"http://{local_ip}:8501"
    custom_url = st.text_input(t("qr_url_label"), value=default_url)
    url_to_encode = custom_url if custom_url.strip() != "" else default_url
    qr = qrcode.QRCode(box_size=4, border=2)
    qr.add_data(url_to_encode)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="#38BDF8", back_color="#0A0F1E")
    buf = BytesIO()
    qr_img.save(buf, format="PNG")
    buf.seek(0)
    st.image(buf, caption=f"🔗 {url_to_encode}", width='stretch')
    
    st.markdown("---")
    st.caption(t("disclaimer"))

# =============================================================================
# 9. MAIN UI (inchangé sauf suppression de st.balloons())
# =============================================================================
if model is None:
    st.error("❌ Erreur critique : impossible de charger le modèle. Vérifiez le fichier 'modele_ml.pkl'.")
    st.stop()

col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    # Logo personnalisé (sans emoji)
    st.markdown("""
    <div class="custom-logo">
        <div class="logo-circle">
            <span>❤️</span>
        </div>
        <div class="logo-text">
            <span class="dia">DiaPredict</span> <span class="ai">AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <h1 class="gradient-text">{t('title')}</h1>
        <p style="color: #94A3B8; margin-top: -0.5rem;">{t('subtitle')}</p>
        <div class="custom-hr"></div>
        <p style="font-size: 0.9rem; color: #CBD5E1; text-align: left;">
            <strong>{t('how_it_works')}</strong><br>
            {t('how_text')}<br><br>
            {t('accuracy')}<br>
            {t('threshold_custom')}
        </p>
        <div style="margin-top: 1rem;">
            <span style="background: rgba(56, 189, 248, 0.2); padding: 0.3rem 1rem; border-radius: 2rem; font-size: 0.7rem;">{t('anon')}</span>
            <span style="background: rgba(168, 85, 247, 0.2); padding: 0.3rem 1rem; border-radius: 2rem; font-size: 0.7rem; margin-left: 0.5rem;">{t('ai_cert')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Petite animation SVG
    st.markdown(f"""
    <div style="margin-top: 2rem; text-align: center;">
        <svg width="200" height="80" viewBox="0 0 200 80" xmlns="http://www.w3.org/2000/svg">
            <path d="M10,60 L40,30 L70,45 L100,20 L130,40 L160,25 L190,50" stroke="#38BDF8" stroke-width="2" fill="none" stroke-dasharray="5,5">
                <animate attributeName="stroke-dashoffset" from="200" to="0" dur="3s" repeatCount="indefinite"/>
            </path>
            <circle cx="10" cy="60" r="3" fill="#F472B6"/>
            <circle cx="190" cy="50" r="3" fill="#A78BFA"/>
        </svg>
        <p style="color:#475569; font-size:0.7rem;">{t('realtime')}</p>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f"<h3 style='margin: 0 0 0.5rem 0; background: linear-gradient(120deg, #A78BFA, #38BDF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{t('clinical_params')}</h3>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input(t("pregnancies"), 0, 20, 0)
        glucose = st.number_input(t("glucose"), 0, 300, 100)
        blood_pressure = st.number_input(t("blood_pressure"), 0, 200, 70)
        skin_thickness = st.number_input(t("skin_thickness"), 0, 100, 20)
    with c2:
        insulin = st.number_input(t("insulin"), 0, 900, 80)
        bmi = st.number_input(t("bmi"), 0.0, 70.0, 25.0, format="%.1f")
        dpf = st.number_input(t("dpf"), 0.0, 3.0, 0.5, format="%.3f")
        age = st.number_input(t("age"), 21, 120, 30)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button(t("predict_button"), width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)
    
    if predict_btn:
        input_df = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }])
        input_df = preprocess_input(input_df)
        
        with st.spinner(t("analyzing")):
            time.sleep(0.8)
            proba = float(model.predict_proba(input_df)[:, 1][0])
            pred = int(proba >= st.session_state.threshold)  # seuil dynamique
        
        st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
        
        # Jauge circulaire
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
            <div style="flex: 1; text-align: center;">
                <div style="position: relative; width: 180px; height: 180px; margin: 0 auto;">
                    <svg viewBox="0 0 200 200">
                        <circle cx="100" cy="100" r="90" fill="none" stroke="#1E293B" stroke-width="12"/>
                        <circle cx="100" cy="100" r="90" fill="none" stroke="url(#grad)" stroke-width="12" 
                                stroke-dasharray="{2 * 3.14159 * 90}" stroke-dashoffset="{2 * 3.14159 * 90 * (1 - proba)}"
                                stroke-linecap="round" transform="rotate(-90 100 100)"/>
                        <defs>
                            <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stop-color="#38BDF8"/>
                                <stop offset="100%" stop-color="#F472B6"/>
                            </linearGradient>
                        </defs>
                        <text x="100" y="110" text-anchor="middle" fill="white" font-size="36" font-weight="bold">{proba*100:.1f}%</text>
                        <text x="100" y="140" text-anchor="middle" fill="#94A3B8" font-size="14">{t('risk')}</text>
                    </svg>
                </div>
            </div>
            <div style="flex: 2; padding-left: 1rem;">
                <div style="background: rgba(0,0,0,0.3); border-radius: 1rem; padding: 0.5rem 1rem;">
                    <p style="margin:0; color:#CBD5E1;">{t('threshold_label')} : <strong>{st.session_state.threshold*100:.0f}%</strong></p>
                    <div style="background: #1E293B; border-radius: 1rem; height: 8px; margin-top: 5px;">
                        <div style="background: linear-gradient(90deg, #38BDF8, #F472B6); width: {st.session_state.threshold*100}%; height: 8px; border-radius: 1rem;"></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if pred == 1:
            st.markdown(f"""
            <div class="result-card" style="background: rgba(239, 68, 68, 0.15); border-left: 5px solid #EF4444;">
                <h4 style="color: #F87171; margin:0;">{t('alert_high')}</h4>
                <p style="color: #FECACA;">{t('alert_high_msg')}</p>
                <p style="font-size: 0.8rem; color: #FDA4AF;">{t('alert_high_advice')}</p>
            </div>
            """, unsafe_allow_html=True)
            # st.balloons()  <--- SUPPRIMÉ
        else:
            st.markdown(f"""
            <div class="result-card" style="background: rgba(74, 222, 128, 0.1); border-left: 5px solid #4ADE80;">
                <h4 style="color: #4ADE80; margin:0;">{t('alert_low')}</h4>
                <p style="color: #BBF7D0;">{t('alert_low_msg')}</p>
                <p style="font-size: 0.8rem; color: #86EFAC;">{t('alert_low_advice')}</p>
            </div>
            """, unsafe_allow_html=True)

        # Explication des facteurs clés: affichée après le diagnostic de risque.
        indicator_analysis = analyze_key_indicators(
            pregnancies=pregnancies,
            glucose=glucose,
            blood_pressure=blood_pressure,
            skin_thickness=skin_thickness,
            insulin=insulin,
            bmi=bmi,
            dpf=dpf,
            age=age,
            language=st.session_state.language,
        )

        st.markdown(f"""
        <div class="result-card" style="background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(56, 189, 248, 0.35);">
            <h4 style="margin: 0; color: #7DD3FC;">{indicator_analysis['title']}</h4>
            <p style="margin: 0.4rem 0 0.8rem 0; color: #CBD5E1; font-size: 0.92rem;">{indicator_analysis['subtitle']}</p>
        </div>
        """, unsafe_allow_html=True)

        cards_cols = st.columns(2)
        for idx, item in enumerate(indicator_analysis["details"]):
            status_color = {
                "low": "#F59E0B",
                "normal": "#4ADE80",
                "elevated": "#F97316",
                "high": "#EF4444",
            }.get(item["status_key"], "#38BDF8")

            with cards_cols[idx % 2]:
                st.markdown(f"""
                <div class="result-card" style="background: rgba(2, 6, 23, 0.5); border-left: 4px solid {status_color}; min-height: 250px;">
                    <p style="margin: 0; color: #E2E8F0; font-weight: 700;">{item['label']}</p>
                    <p style="margin: 0.3rem 0; color: #94A3B8; font-size: 0.82rem;">{item['what']}</p>
                    <p style="margin: 0.3rem 0 0.15rem 0; color: #CBD5E1;"><strong>{item['value']} {item['unit']}</strong></p>
                    <span style="display:inline-block; background: rgba(148, 163, 184, 0.16); color:{status_color}; padding: 0.15rem 0.55rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700;">{item['status']}</span>
                    <p style="margin-top: 0.7rem; color: #CBD5E1; font-size: 0.82rem;"><strong>{item['why_label']}:</strong> {item['why']}</p>
                    <p style="margin-top: 0.4rem; color: #A7F3D0; font-size: 0.8rem;"><strong>{item['advice_label']}:</strong> {item['advice']}</p>
                </div>
                """, unsafe_allow_html=True)

        explain_col1, explain_col2 = st.columns(2)
        with explain_col1:
            up_items = indicator_analysis["up_factors"] or [indicator_analysis["none_up"]]
            up_html = "".join([f"<li>{x}</li>" for x in up_items])
            st.markdown(f"""
            <div class="result-card" style="background: rgba(239, 68, 68, 0.1); border-left: 4px solid #EF4444;">
                <p style="margin:0; color:#FCA5A5; font-weight:700;">{indicator_analysis['up_title']}</p>
                <ul style="margin:0.4rem 0 0 1rem; color:#FECACA; font-size:0.86rem;">{up_html}</ul>
            </div>
            """, unsafe_allow_html=True)

        with explain_col2:
            down_items = indicator_analysis["down_factors"] or [indicator_analysis["none_down"]]
            down_html = "".join([f"<li>{x}</li>" for x in down_items])
            st.markdown(f"""
            <div class="result-card" style="background: rgba(74, 222, 128, 0.1); border-left: 4px solid #4ADE80;">
                <p style="margin:0; color:#86EFAC; font-weight:700;">{indicator_analysis['down_title']}</p>
                <ul style="margin:0.4rem 0 0 1rem; color:#DCFCE7; font-size:0.86rem;">{down_html}</ul>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style="text-align: center; margin-top: 3rem; padding: 1rem;">
    <hr style="background: linear-gradient(90deg, transparent, #38BDF8, transparent); border: none; height: 1px;">
    <p style="color: #334155; font-size: 0.75rem;">{t('footer')}</p>
</div>
""", unsafe_allow_html=True)