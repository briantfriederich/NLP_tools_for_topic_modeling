'''
Creator: Chane Jackson
Date: 02 May 2019
'''

#Import necessary libraries
import os
import textract
import PySimpleGUI as sg
import pdb

class TextExtraction():
    def __init__(self):
        '''This function will create the class for extraction:
        - Initialize the folder for the TextExtraction
        - Functions for looping through the file folders
        - Function for extracting the texts
        - Function for saving each text extracted as a text file
        '''
        #Prompt user with simple GUI for path
        event, (filePath,) = sg.Window('Get Directory Path').Layout([[sg.Text('Path')],[sg.Input(), sg.FolderBrowse()], [sg.OK(), sg.Cancel()]]).Read()

        #Save the given path
        self.path = filePath
        self.pathNew = self.path+"/textFiles"

        #Create a new folder
        try:
            os.mkdir(self.pathNew)
        except OSError:
            print("Creation of the directory %s failed" % self.pathNew)
        else:
            print("Successfully created the directory %s" % self.pathNew)

    def getFiles(self):
        '''This function will take each file & turn it into text a text document
        - Loop through the directory
        - Extract the text from the files
        - Save the text into a text document in the new folder
        '''
        #Initiate the loop through the directory
        for root, dirs, files in os.walk(self.path):
            print("Directory: {}\nFiles:{}".format(dirs, files))
            counter = 0
            stopper = len(files)
            for file in files:
                counter = counter + 1
                #Remove the extension
                filename, file_extension = os.path.splitext(file)
                print("Filename: {}\tFile Extension: {}".format(filename, file_extension))
                try:
                    if (file_extension != '') and (counter < stopper + 1):
                        #Save the filename
                        filename = filename +'.txt'

                        #Extract the text from the document
                        print("Path: {}\tFile: {}\n".format(self.path, file))
                        print(textract.process(r'C:\Users\user\Desktop\NLP_tools_for_topic_modeling\sample_corpus\arabic_doc.txt'))
                        tempText = textract.process(self.path+'/'+file)
                        print(f"tempText: {tempText}")

                        #Save the given text into a text document
                        self.saveFile(tempText, filename)
                        print(f"File Saved: {filename}")
                    elif file_extension == '':
                        print("%s is not an actual file" % filename)
                    print("_______________________\n")
                    #elif counter == stopper:
                        #break
                except: textract.exceptions.MissingFileError





    def saveFile(self, txtDoc,fileName):
        '''This function will take the text extracted bytes and save it to a file_extension
        - Change text to string
        - Put into text file
        '''
        #Change text extracted bytes into a string
        txtDoc = str(txtDoc)

        #Create the append or write parameter
        if os.path.exists(self.pathNew+'/'+fileName):
            append_write = 'a'
        else:
            append_write = 'w+'

        doc = open(self.pathNew+'/'+fileName, append_write)
        doc.write(txtDoc)
        doc.close()






if __name__ == "__main__":
    #Establish Class TextExtraction
    root = "C:/Users/user/Desktop/"
    c = TextExtraction()

    #Run the program to get the text files
    c.getFiles()
