import io
import os
import PyPDF2

folder = (r"C:\Users\user\Desktop\NLP_tools_for_topic_modeling\sample_corpus\تنظيم عقد.pdf")

"""
Extract PDF text using PDFMiner. Adapted from
http://stackoverflow.com/questions/5725278/python-help-using-pdfminer-as-a-library
"""
import pdfminer as pdfminer
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter#process_pdf
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams

from io import StringIO

def pdf_to_text(pdfname):

    # PDFMiner boilerplate
    rsrcmgr = PDFResourceManager()
    sio = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, sio, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # Extract text
    fp = open(pdfname, 'rb', encoding = 'utf8')
    for page in PDFPage.get_pages(fp):
        print(page)
        interpreter.process_page(page)
        print(interpreter.process_page(page))
    fp.close()

    # Get text from StringIO
    text = sio.getvalue()
    print(f"Text: {text}")

    # Cleanup
    device.close()
    sio.close()

    return text

print(pdf_to_text(folder))