import os
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
    fp = open(pdfname, 'rb')
    for page in PDFPage.get_pages(fp):
        #print(page)
        interpreter.process_page(page)
        #print(interpreter.process_page(page))
    fp.close()

    # Get text from StringIO
    #print(len(sio.getvalue()))
    text = sio.getvalue()
    #print(f"Text: {text}")
    text=text.replace(chr(272)," ")
    #print(type(text))

    # Cleanup
    device.close()
    sio.close()

    return text


print("enter directory path: ")
directory_path = input()

if __name__ == "__main__":

    pathNew = os.path.join(os.environ["HOMEPATH"], "Desktop") + "\\textFiles"

    #Create a new folder
    try:
        os.mkdir(pathNew)
    except OSError:
        print("Creation of the directory %s failed" % pathNew)
    else:
        print("Successfully created the directory %s" % pathNew)

    for path, dirs, files in os.walk(directory_path):
        for filename in files:
            short_filename, file_extension = os.path.splitext(filename)
            if file_extension == '.pdf':
                #print(f"attempting to process {short_filename}")
                try:
                    doctext = pdf_to_text(directory_path +'\\' + filename)
                    print(f"len doc: {len(doctext)}")
                    if len(doctext) > 50:
                        #print(f"attempting to process {short_filename}")
                        #print(f"len doc: {len(doctext)}")
                        #print(f"doctext: {doctext}\n\n")
                        doc = open(pathNew +'\\' + short_filename + '.txt', "w+")
                        doc.write(doctext)
                        doc.close()
                        print(f"{short_filename} written to file\n")
                    else:
                        print(f"{short_filename} could not be processed\n")
                except:
                    #print(f"{short_filename} could not be opened\n")
                    continue

        
