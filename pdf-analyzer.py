#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: David Espejo (Fortytwo Security)

import argparse
import PyPDF2
from gpt4all import GPT4All
from transformers import GPT2Tokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text.
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf = PyPDF2.PdfFileReader(file)
            text = ''
            for page in range(pdf.getNumPages()):
                text += pdf.getPage(page).extractText()
        return text
    except FileNotFoundError:
        logging.error(f"File not found: {pdf_path}")
        return ''
    except PyPDF2.utils.PdfReadError as e:
        logging.error(f"Error reading PDF file: {e}")
        return ''

def analyze_text_with_gpt4all(text, model_name, prompt):
    """
    Analyze text using GPT4All.

    Args:
        text (str): The text to analyze.
        model_name (str): The name of the GPT4All model.
        prompt (str): The user prompt.

    Returns:
        str: The generated response.
    """
    try:
        # Initialize GPT4All with the model name
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
        truncated_prompt = tokenizer.decode(tokens, clean_up_tokenization_spaces=True)

        # Generate a response
        response = gpt.generator(truncated_prompt)

        return response
    except Exception as e:
        logging.error(f"Error analyzing text with GPT4All: {e}")
        return "An error occurred while generating the response."

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Analyze a PDF file with GPT4All')
    parser.add_argument('--pdf', '-f', type=str, required=True, help='Path to the PDF file')
    parser.add_argument('--model', '-m', type=str, required=True, help='GPT4All model name')
    parser.add_argument('--prompt', '-p', type=str, required=True, help='User prompt')

    args = parser.parse_args()

    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(args.pdf)
    if not pdf_text:
        logging.error("Failed to extract text from PDF.")
        return

    # Analyze the text with GPT4All
    response = analyze_text_with_gpt4all(pdf_text, args.model, args.prompt)

    # Print the response
    print(response)

if __name__ == "__main__":
    main()
