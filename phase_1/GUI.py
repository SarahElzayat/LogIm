import customtkinter
from PIL import Image, ImageTk
import os
# from ipynb.fs.full.final_integration import solve_expression

customtkinter.set_appearance_mode("dark")

root = customtkinter.CTk()
root.geometry("900x500")

string = " "

def browser():
    filename=customtkinter.filedialog.askopenfilename(initialdir=os.getcwd(), filetype=(("JPG File","*.jpg"),("PNG File","*.png")))
    img=Image.open(filename)
    img = img.resize((300,300))
    img=ImageTk.PhotoImage(img)
    lbl.configure(image=img)
    lbl.image=img
    # string = solve_expression(img)
    # string = "result"

def convert():
    lbl2.configure(text = "result") # string

frame = customtkinter.CTkFrame(master = root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

lbl=customtkinter.CTkLabel(frame, text="")
lbl.place(relx=0.7, rely=0.4, anchor=customtkinter.CENTER)

btn = customtkinter.CTkButton(frame, text="<=", command=convert, height= 20, width=20)
btn.place(relx=0.5, rely=0.5, anchor=customtkinter.CENTER)

lbl2=customtkinter.CTkLabel(frame, text="")
lbl2.place(relx=0.3, rely=0.4, anchor=customtkinter.CENTER)

btn = customtkinter.CTkButton(frame, text="select image", command=browser)
btn.place(relx=0.45, rely=0.9, anchor=customtkinter.SE)

btn = customtkinter.CTkButton(frame, text="exit", command=lambda:exit())
btn.place(relx=0.55, rely=0.9, anchor=customtkinter.SW)

root.mainloop()