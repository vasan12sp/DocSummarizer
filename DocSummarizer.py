import streamlit as st
import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification, BartTokenizer, BartForConditionalGeneration
import math
import numpy as np
from PyPDF2 import PdfReader

# Model Class
class Summ_xlnet_bart:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load XLNet
        self.xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.xlnet_model = XLNetForSequenceClassification.from_pretrained(
            'xlnet-base-cased',
            num_labels=1
        ).to(self.device)

        # Load BART
        self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.bart_model = BartForConditionalGeneration.from_pretrained(
            'facebook/bart-large-cnn'
        ).to(self.device)

        # Set models to evaluation mode
        self.xlnet_model.eval()
        self.bart_model.eval()

    @torch.no_grad()
    def summarize_chunk(self, paragraph, target_abstractive_percent=0.6):
        original_word_count = len(paragraph.split())
        target_abstractive_words = math.ceil(original_word_count * target_abstractive_percent)
        extractive_percent = min(0.9, target_abstractive_percent * 1.5)

        sentences = [s.strip() for s in paragraph.split('.') if s.strip()]
        num_sentences = len(sentences)

        if num_sentences <= 2:
            return paragraph

        sentences_to_keep = max(2, math.ceil(num_sentences * extractive_percent))
        inputs = self.xlnet_tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(self.device)

        outputs = self.xlnet_model(**inputs)
        scores = outputs.logits.squeeze().cpu().numpy()

        ranked_indices = np.argpartition(scores, -sentences_to_keep)[-sentences_to_keep:]
        original_order_indices = sorted(ranked_indices)
        extractive_summary = '. '.join(sentences[i] for i in original_order_indices) + '.'

        target_tokens = math.ceil(target_abstractive_words * 1.3)
        min_length = max(10, math.ceil(target_tokens * 0.9))
        max_length = min(math.ceil(target_tokens * 1.1), 1024)

        bart_input = self.bart_tokenizer(
            extractive_summary,
            return_tensors='pt',
            max_length=1024,
            truncation=True
        ).to(self.device)

        bart_output = self.bart_model.generate(
            **bart_input,
            min_length=min_length,
            max_length=max_length,
            length_penalty=1.5,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2
        )

        return self.bart_tokenizer.decode(bart_output[0], skip_special_tokens=True)

    def summarize_large_text(self, text, target_abstractive_percent=0.6, chunk_size=512):
        chunks = self.chunk_text(text, chunk_size)
        summaries = []
        for chunk in chunks:
            summaries.append(self.summarize_chunk(chunk, target_abstractive_percent))
        return ' '.join(summaries)

    def chunk_text(self, text, chunk_size):
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    @classmethod
    @st.cache_resource
    def load_model(cls, path):
        instance = cls()
        model_state = torch.load(path, map_location=instance.device)
        instance.xlnet_model.load_state_dict(model_state['xlnet_state'])
        instance.bart_model.load_state_dict(model_state['bart_state'])
        return instance

# Caching the PDF reading function
@st.cache_data
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Streamlit UI
st.title("Document Summarizer")

# Model loading with caching
model_path = "/home/vasan12sp/Downloads/summ_xlnet_bart.pth"  
model_instance = Summ_xlnet_bart.load_model(model_path)

# Input options for file upload or text input
upload_file = st.file_uploader("Upload a PDF", type=["pdf"])
text_input = st.text_area("Or paste a paragraph below:", "")

# Abstractive percentage selection using a slider
target_abstractive_percent = st.slider(
    "Select the percentage of summarization (abstractive reduction):", 
    min_value=30, 
    max_value=80, 
    value=60, 
    step=10
) / 100  # Convert slider value to a fraction

# Summarize button
if st.button("Summarize"):
    if upload_file is not None:
        input_text = read_pdf(upload_file)
        st.write("Summarizing uploaded PDF...")
    elif text_input.strip() != "":
        input_text = text_input
        st.write("Summarizing pasted text...")
    else:
        st.warning("Please upload a PDF or paste a paragraph.")
        input_text = None

    if input_text:
        # Handle large text by chunking
        summary = model_instance.summarize_large_text(input_text, target_abstractive_percent)
        st.subheader("Summary:")
        st.write(summary)
