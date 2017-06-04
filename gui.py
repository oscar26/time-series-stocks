#import Tkinter as *
from Tkinter import *

root= Tk()

topFrame=Frame(root)
topFrame.pack()

bottomFrame=Frame(root)
bottomFrame.pack(side=BOTTOM)

button1=Button(topFrame, text="Aceptar 1",fg="red");
button2=Button(topFrame, text="Aceptar 2",fg="blue");
button3=Button(topFrame, text="Aceptar 3",fg="green");
button4=Button(bottomFrame, text="Aceptar 4",fg="red");

button1.pack(side=LEFT);
button2.pack(side=LEFT);
button3.pack(side=LEFT);
button4.pack(side=BOTTOM);


root.mainloop()
