# NLP ------------------------------------------------------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModel

def get_language_type(domain):
    if domain in ["M2M-M", "M2M-R"]:
        return "en"
    elif domain in ["Multilingual-en", "Multilingual-en", "Multilingual-th"]:
        return "multilingual"
    else:
        return "kor" # weather, navi

def get_text_reader(domain, reader_name):
    lang_type = get_language_type(domain)

    if lang_type == "en":
        bert_name = "bert-base-uncased"
        text_reader = AutoModel.from_pretrained(bert_name)

    elif lang_type == "multilingual":
        bert_name = "bert-base-multilingual-cased"
        text_reader = AutoModel.from_pretrained(bert_name)

    else: # kor
        if reader_name == "kobert":
            from transformers import BertModel
            # bert_name = "monologg/kobert"
            # text_reader = BertModel.from_pretrained(bert_name)
            raise NotImplementedError("Kobert is not supported in this version.")
        else: # multilingual-bert
            bert_name = "bert-base-multilingual-cased"
            text_reader = AutoModel.from_pretrained(bert_name)

    return text_reader

def get_tokenizer(domain, reader_name):
    lang_type = get_language_type(domain)

    if lang_type == "en":
        bert_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(bert_name)

    elif lang_type == "multilingual":
        bert_name = "bert-base-multilingual-cased"
        tokenizer = AutoTokenizer.from_pretrained(bert_name)

    else:  # kor
        if reader_name == "kobert":
            # from utils.tokenization_kobert import KoBertTokenizer
            # tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
            raise NotImplementedError("Kobert is not supported in this version.")
        else:  # multilingual-bert
            bert_name = "bert-base-multilingual-cased"
            tokenizer = AutoTokenizer.from_pretrained(bert_name)

    return tokenizer
# ----------------------------------------------------------------------------------------------------------------------


# Vision ---------------------------------------------------------------------------------------------------------------
def get_image_reader(reader_name, num_labels):
    if reader_name == "cnn":
        from cnn import ImageEncoder
        image_reader = ImageEncoder(num_labels)

    elif reader_name == "resnet":
        raise NotImplementedError("Resnet is not supported in this version.")

    elif reader_name == "vgg":
        raise NotImplementedError("Vgg is not supported in this version.")

    else:
        raise KeyError(reader_name)

    return image_reader
# ----------------------------------------------------------------------------------------------------------------------