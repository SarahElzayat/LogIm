import customtkinter
from PIL import Image, ImageTk
import os
from ttg import Truths
# import tkinter as tk

# class Table:
     
#     def __init__(self,root):
         
#         # code for creating table
#         for i in range(total_rows):
#             for j in range(total_columns):  
#                 self.e = tk.Entry(root, fg='blue',width="2",font=('Arial',16,'bold'))
#                 self.e.place(x=80+30*i, y=80+30*j)
#                 self.e.insert(tk.END, lst[i][j])
#                 self.e.config(state= "disabled")

# lst = Truths(['A', 'B', 'C','D'], ['A and B and C and D']).as_pandas().values.T.tolist()

# total_rows = len(lst)
# total_columns = len(lst[0])


customtkinter.set_appearance_mode("dark")

root = customtkinter.CTk()
root.geometry("900x500")

def browser():
    filename=customtkinter.filedialog.askopenfilename(initialdir=os.getcwd(), filetype=(("JPG File","*.jpg"),("PNG File","*.png")))
    img=Image.open(filename)
    img = img.resize((250,400))
    img=ImageTk.PhotoImage(img)
    lbl.configure(image=img)
    lbl.image=img
    lbl.grid(row=3, column=4, padx=10, pady=10,columnspan=2,rowspan=2)

def convert():
    text = '''
    +-----+-----+-----+-----+-----------------------+
    |  A  |  B  |  C  |  D  |  A and B and C and D  |
    |-----+-----+-----+-----+-----------------------|
    |  1  |  1  |  1  |  1  |           1           |
    |  1  |  1  |  1  |  0  |           0           |
    |  1  |  1  |  0  |  1  |           0           |
    |  1  |  1  |  0  |  0  |           0           |
    |  1  |  0  |  1  |  1  |           0           |
    |  1  |  0  |  1  |  0  |           0           |
    |  1  |  0  |  0  |  1  |           0           |
    |  1  |  0  |  0  |  0  |           0           |
    |  0  |  1  |  1  |  1  |           0           |
    |  0  |  1  |  1  |  0  |           0           |
    |  0  |  1  |  0  |  1  |           0           |
    |  0  |  1  |  0  |  0  |           0           |
    |  0  |  0  |  1  |  1  |           0           |
    |  0  |  0  |  1  |  0  |           0           |
    |  0  |  0  |  0  |  1  |           0           |
    |  0  |  0  |  0  |  0  |           0           |
    +-----+-----+-----+-----+-----------------------+
    
            '''
    lbl2.configure(text)
    return

def get_value():
    """ returns selected value as a string, returns an empty string if nothing selected """
    return radio_button_var.get()

def set_value(selection):
    """ selects the corresponding radio button, selects nothing if no corresponding radio button """
    radio_button_var.set(selection)

frame = customtkinter.CTkFrame(master = root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

# t = Table(frame)

btn = customtkinter.CTkButton(frame, text="select image", command=browser)
btn.grid(row=1, column=1, padx=10, pady=10)

btn = customtkinter.CTkButton(frame, text="exit", command=lambda:exit())
btn.grid(row=1, column=2, padx=10, pady=10)

btn = customtkinter.CTkButton(frame, text="convert", command=convert)
btn.grid(row=1, column=3, padx=10, pady=10)

lbl2=customtkinter.CTkLabel(frame, text="")

lbl=customtkinter.CTkLabel(frame, text="")

radio_button_var = customtkinter.StringVar(value="t2e")
radio_button_1 = customtkinter.CTkRadioButton(frame, text="table to expression", value="t2e", variable=radio_button_var)
radio_button_1.grid(row=1, column=4, padx=10, pady=10,columnspan=2)
radio_button_2 = customtkinter.CTkRadioButton(frame, text="expression to table", value="e2t", variable=radio_button_var)
radio_button_2.grid(row=2, column=4, padx=10, pady=10,columnspan=2)

progressbar = customtkinter.CTkProgressBar(master=frame,orientation='vertical',width=5,height=500)
progressbar.set(1)
progressbar.place(relx=0.5, rely=0.75, anchor=customtkinter.CENTER)

root.mainloop()