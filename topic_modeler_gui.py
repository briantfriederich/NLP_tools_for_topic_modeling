import os
from tkinter import *
#from tkinter import tkk
from tkinter.ttk import *
from tkinter import messagebox
from tkinter.ttk import Progressbar
from tkinter import filedialog
from tkinter import Menu



 
window = Tk()
window.title("Topic Modeler")
window.geometry('650x175')


def help_popup():
    messagebox.showinfo("How to Use Tool", "a Tk MessageBox")

def info_popup():
    messagebox.showinfo("Algorithms Info", "a Tk MessageBox")


def clicked():
    try:
        corpus_folder = txt.get()
        assert corpus_folder != ""
        perform_algs(corpus_folder)
    except:
        messagebox.showerror('Error', 'Please input folder of documents into text field')

def perform_algs(content):
    try:
        assert (selected.get()) > 0
        print(content)
        if selected.get() == 1:
            print("do tfidf")
        else:
            print("do LDA")
    except:
        messagebox.showerror('Error', 'Please select algorithm to perform')


content = ''
filename = ''

def open_file():
    global content
    global file_path

    filename = filedialog.askopenfilename()
    infile = open(filename, 'r')
    content = infile.read()
    file_path = os.path.dirname(filename)
    entry.delete(0, END)
    entry.insert(0, file_path)
    return filename, content

def process_file(filename, content):
    try:
        assert len(file_path) > 0
        perform_algs(content)
    except:
        messagebox.showerror('Error', 'Please input folder of documents into text field')

file_path = StringVar
selected = IntVar()

menu = Menu(window)
new_item = Menu(menu)
new_item.add_command(label='How to Use Tool', command=help_popup)
new_item.add_command(label='Algorithms Info', command=info_popup)
menu.add_cascade(label='Help', menu=new_item)
window.config(menu=menu)

Label(text="Select Your File (Only txt files)").grid(row=0, column=0, sticky='w')
entry = Entry(width=50, textvariable=file_path)
entry.grid(row=0,column=1,padx=2,pady=2,sticky='we',columnspan=25)
Button(text="Browse", command=open_file).grid(row=0, column=27, sticky='ew', padx=8, pady=4)


Label(window, text="Select Algorithm").grid(row=1, column=0, sticky='w') 
radiobutton1 = Radiobutton(window,text='Document Snowflake', value=1, variable=selected)
radiobutton1.grid(row=1, column=1, padx = 8)
radiobutton2 = Radiobutton(window,text='Latent Topic Grouper', value=2, variable=selected)
radiobutton2.grid(row=1, column=2, padx = 8)


Button(text="Process Now", width=32, command=lambda: process_file(filename, content)).grid(row=2, column=1, sticky='ew', pady=10)

Label(window, text="Progress").grid(row=5, column=0, sticky='w', pady=7)
bar = Progressbar(window, length=300, value=70).grid(row=5, column=1, sticky='w', columnspan=27) 

window.mainloop()