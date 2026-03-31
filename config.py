# 1. HAKEM MODEL (Kaliteyi Ölçen)
QUALITY_MODEL = 'buffalo_l'

# 2. İŞÇİ MODEL (Kimlik Eşleştiren) - Test etmek istediğin modele göre burayı değiştireceksin
MATCHER_MODEL = 'antelopev2' # İlk testimizi zayıf modelle (Zarar Vermeme Testi) yapacağız

DECISION_THRESHOLD = 0.25

# ==========================================
# 📁 DİNAMİK DOSYA YOLLARI (BURAYA DOKUNMA)
# ==========================================
LFW_BASE_CACHE = f"lfw_base_embeddings_{QUALITY_MODEL}.pkl"
LFW_TRAIN_DATA = f"lfw_train_data_{QUALITY_MODEL}_thresh{DECISION_THRESHOLD}.pkl"
CLASSIFIER_PATH = f"classifier_{QUALITY_MODEL}_thresh{DECISION_THRESHOLD}.pth"

BAYES_MODEL_PATH = f"tinyface_bayes_model_{MATCHER_MODEL}.pkl"

QUALITY_CACHE = f"ijbc_embeddings_cache_{QUALITY_MODEL}.pkl"
MATCHER_CACHE = f"ijbc_embeddings_cache_{MATCHER_MODEL}.pkl"

OUTPUT_ROC_PLOT = f"ijbc_adaptive_ROC_{MATCHER_MODEL}_with_{QUALITY_MODEL}.png"
OUTPUT_EDC_PLOT = f"ijbc_adaptive_EDC_{MATCHER_MODEL}_with_{QUALITY_MODEL}.png"