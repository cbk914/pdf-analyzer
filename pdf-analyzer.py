#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: David Espejo (Fortytwo Security)
import argparse
import PyPDF2
from gpt4all import GPT4All
from transformers import GPT2Tokenizer

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf = PyPDF2.PdfFileReader(file)
        text = ''
        for page in range(pdf.getNumPages()):
            text += pdf.getPage(page).extractText()
    return text

# Function to analyze text with GPT4All
def analyze_text_with_gpt4all(text, model_name, prompt):
    # Initialize GPT4All with a model name
    gpt = GPT4All(model_name)

    # Initialize GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Combine the extracted text and the user prompt
    full_prompt = text + '\n' + prompt

    # Tokenize the full prompt
    tokens = tokenizer.encode(full_prompt)

    # Check if the tokenized full prompt exceeds the model's token limit
    if len(tokens) > 4096:
        # Truncate the tokens to the model's token limit
        tokens = tokens[:4096]

    # Decode the tokens back into text
    truncated_prompt = tokenizer.decode(tokens)

    # Generate a response
    response = gpt.generator(truncated_prompt)

    return response

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Analyze a PDF file with GPT4All')
parser.add_argument('--pdf', '-f', type=str, required=True, help='Path to the PDF file')
parser.add_argument('--model', '-m', type=str, required=True, help='GPT4All model name')
parser.add_argument('--prompt', '-p', type=str, required=True, help='User prompt')

args = parser.parse_args()

# Extract text from the PDF
pdf_text = extract_text_from_pdf(args.pdf)

# Analyze the text with GPT4All
response = analyze_text_with_gpt4all(pdf_text, args.model, args.prompt)

# Print the response
print(response)
