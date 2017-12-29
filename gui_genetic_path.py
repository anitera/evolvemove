from tkinter import *
import numpy as np

class GUI(Frame):
    def __init__(self, coord):
        super().__init__()
        self.coord = coord
        self.initUI()

    def initUI(self):
        self.master.title("Colours")
        self.pack(fill=BOTH, expand=1)

        canvas = Canvas(self)
        canvas.create_rectangle(0, 0, 320, 380,
                                outline="#ba0", fill="#ba0")
        for i in range(self.coord.shape[1]):
            canvas.create_rectangle(self.coord[:, i][0], self.coord[:, i][1], self.coord[:, i][0]+3, self.coord[:, i][1]+3,
                                    outline="#f50", fill="#f50")
        canvas.pack(fill=BOTH, expand=1)

def main(vector):
    root = Tk()
    ex = GUI(vector)
    root.geometry("900x600")

    e = Entry(root)
    e.pack()

    e.focus_set()

    def callback():
        print(e.get())

    b = Button(root, text="Parameter 1", width=10, command=callback)
    b.pack()

    e1 = Entry(root)
    e1.pack()

    e1.focus_set()

    def callback1():
        print(e1.get())

    b1 = Button(root, text="Parameter 2", width=10, command=callback1)
    b1.pack()

    root.mainloop()

if __name__ == '__main__':
    vector_coord = np.array([[10,20,30,40,50,60,70,80], [25, 34, 56, 72, 12, 45, 23, 56]])
    main(vector_coord)