import glob
import PyPDF2



from docx import Document 
document = Document('C:\Users\user\Desktop\REC_topic_modeling\DOCS\Patch 4')

print(document.paragraphs)


'''

class TextImporter():
	def __init__(self, path):
		self.path = path

	def text_import(self):
	    files = glob.glob(self.path + '*')
	    documents=list()
	    for textfile in files[:]:
	        if '.txt' in textfile:
	            textfile_open = open(textfile,'r')
	            textfileReader = textfile_open.read()
	            textfile_1 = ''
	            textfile_1 = textfileReader
	            documents.append(textfile_1)
	        elif '.pdf' in textfile:
	            pdfFileObj = open(textfile, 'rb')
	            pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
	            textfile_1 = ''
	            for page in range(0,pdfReader.numPages):
	                textfile_1 += pdfReader.getPage(page).extractText()
	            documents.append(textfile_1)
	        else:
	            print('File Error: Cannot read file type for ' + textfile + '\n') 
	    return documents



if __name__ == "__main__":


    print("Input path: ")
    path1 = input()
    textextractor = TextImporter(path1)
    print(TextImporter.text_import(textextractor))
    '''