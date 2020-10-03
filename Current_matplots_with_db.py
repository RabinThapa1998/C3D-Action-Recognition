import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)



def plot_graph(win_frame,btn_text):
    btn_text.set("refresh")
    conn = sqlite3.connect('action.db')

    everyday_count_1 = []
    everyday_count_2 = []
    days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    c = conn.cursor()
    for day in days:
        quer1 = f"SELECT COUNT(gun_action) FROM guntable WHERE day='{day}'"
        quer2 = f"SELECT COUNT(fight_action) FROM fighttable WHERE day='{day}'"
        c.execute(quer1)
        everyday_count_1.append(c.fetchall())
        c.execute(quer2)
        everyday_count_2.append(c.fetchall())

    count_daywise_1 = []
    [count_daywise_1.append(count[0][0]) for count in everyday_count_1]

    count_daywise_2 = []
    [count_daywise_2.append(count[0][0]) for count in everyday_count_2]
    # print(count_daywise_1)
    # print(count_daywise_2)

    conn.close()

    x=np.arange(len(days))
    width = 0.23
    fig = Figure(figsize = (8, 4),dpi = 100)
    ax = fig.add_subplot(111)
    rects1 = ax.bar(x - width/2, count_daywise_1, width, label='gun')
    rects2 = ax.bar(x + width/2, count_daywise_2, width, label='fight')
    ax.set_ylabel('total_activities_in_a day')
    ax.set_title('number of actions per day')
    ax.set_xticks(x)
    ax.set_xticklabels(days)
    ax.legend()

    canvas = FigureCanvasTkAgg(fig,master = win_frame)
    canvas.get_tk_widget().delete('all')
    canvas.draw()
    # canvas.get_tk_widget().update()
    canvas.get_tk_widget().pack()

#
# def plot_btn_initiator():
#     plot_graph(window)

# window = Tk()
# window.title('Plotting in Tkinter')
# window.geometry("700x500")
#
# btn_text = StringVar()
# btn_text.set("original_btn")
# plot_button = Button(master=window,
#                      command=plot_btn_initiator,
#                      height=2,
#                      width=10,
#                      textvariable=btn_text)
#
# print( "dsss",plot_button.cget('text'))
# plot_button.pack()
# window.mainloop()
