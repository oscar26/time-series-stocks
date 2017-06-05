# -*- coding: utf-8 -*-
# -*- coding: 850 -*-
import  unicodedata
from Tkinter import *

from parameterGUI import ParameterGUI

root= Tk()


def doNothing():
    print("OK OK I won't.....")


# ******************+
# Configurar Vetana
root.geometry("1000x600+0+0")
root.configure(bg="blue")

# fondo

filename = PhotoImage(file = "./images/fondo5.png")
background_label = Label(root, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


# *******   Menu
menu= Menu(root)
root.config(menu=menu)
subMenu=Menu(menu)




menu.add_cascade(label="File",menu=subMenu)
subMenu.add_command(label="New Prediction",command=doNothing)
subMenu.add_separator()
subMenu.add_command(label="Save",command=doNothing)
subMenu.add_command(label="Save As",command=doNothing)
subMenu.add_separator()
subMenu.add_command(label="Exit",command=root.quit)


editMenu=Menu(menu)
menu.add_cascade(label="Edit",menu=editMenu)
editMenu.add_command(label="Comment",command=doNothing)
editMenu.add_command(label="Change Color ",command=doNothing)
editMenu.add_separator()
editMenu.add_command(label="Undo ",command=doNothing)

# ***********  TOOLBAR  ******
toolbar=Frame(root,bg="#78909C")

msn_parameter=Label(toolbar,bg="#1E88E5", text="Parámetros: ")

msn_parameter.pack(side=LEFT,fill=Y)


toolbar.pack(side=TOP,fill=X)




#  ************************************+
#           Parametros de Entrada
#  ************************************

# Empresas
OPTIONS = ['AAPL','AMZN','BABA','BAC','FB','NKE','NVDA','TSLA','IBM','MSFT']
OPTIONS_MONTH = ['ENERO','FEBRERO','MARZO','ABRIL','MAYO','JUNIO','JULIO','AGOSTO','SEPTIEMBRE','OCTUBRE','NOVIEMBRE','DICIEMBRE']
OPTIONS_DAY = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

isSelectBusinness = StringVar(root)
isSelectBusinness.set(OPTIONS[5]) #  valor por defecto

isSelectMonth = StringVar(root)
isSelectMonth.set(OPTIONS_MONTH[5]) #  valor por defecto

isSelectDay = StringVar(root)
isSelectDay.set(OPTIONS_DAY[0]) #  valor por defecto

select_listBusiness = apply(OptionMenu, (toolbar, isSelectBusinness) + tuple(OPTIONS))


select_month = apply(OptionMenu, (toolbar, isSelectMonth) + tuple(OPTIONS_MONTH))

select_day= apply(OptionMenu, (toolbar, isSelectDay) + tuple(OPTIONS_DAY))


label_name_business=Label(toolbar, text="    Nombre de la Empresa? ")
label_date_pediction=Label(toolbar, text="   Fecha que desea Predecir? ")
label_year_pediction=Label(toolbar,  text=" Año: ")
label_month_pediction=Label(toolbar, text="   Mes:    ")
label_day_pediction=Label(toolbar, text="      Dia:     ")

entry_year_pediction=Entry(toolbar)

label_name_business.pack(side=LEFT, padx=1 ,fill=Y)
select_listBusiness.pack(side=LEFT,padx=1,fill=Y )

label_date_pediction.pack(side=LEFT,fill=Y)

label_year_pediction.pack(side=LEFT,fill=Y)
entry_year_pediction.pack(side=LEFT,fill=Y)

label_month_pediction.pack(side=LEFT,fill=Y)
select_month.pack(side=LEFT,padx=1,fill=Y )

label_day_pediction.pack(side=LEFT,fill=Y)
select_day.pack(side=LEFT,padx=1,fill=Y )


# ouput
def activateCanvas(value_prediction):
    canvas = Canvas(width=300, height=300, bg='white')
    canvas.pack(expand=YES, fill=BOTH, padx=300, pady=200)
    widget = Label(canvas, text='Cierre de la Bolsa es de :', fg='white', bg='black')
    widget.pack(padx=40, pady=30)
    widget = Label(canvas, text=str(value_prediction)+' Dolares', fg='white', bg='red')
    widget.pack()


def beginPrediction(event):

    business = isSelectBusinness.get()
    year = entry_year_pediction.get()
    month = isSelectMonth.get()
    day = isSelectDay.get()

    print  "Empresa ",business
    print  "Año ", year
    print  "Mes ", month
    print  "Dia ", day

    print("Series de Tiempo!")

    parameterGUI = ParameterGUI()
    value_prediction=parameterGUI.beginAnlysis(business,year,month,day)
    activateCanvas(value_prediction)



button_1=Button(toolbar, text="         Empezar          ", bg="#A1887F")
button_1.bind("<Button-1>",beginPrediction)
button_1.pack(side=RIGHT,fill=Y)

# *******************************
# ***********  Status BAR  ******
# *******************************
status=Label(root, text="Universidad Nacional de Colombia",bd=1,relief=SUNKEN,anchor=W)
status.pack(side=BOTTOM,fill=X)
root.mainloop()









root.mainloop()
