import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
from evaluate import evaluate
from tkinter import messagebox
from history import ImageSwitcher
import tkinter.font as tkfont
import os
from draw_colormap import create_color_map_image


def mk_dir():
    if not os.path.exists('./user_results'):
        os.makedirs('./user_results')
    if not os.path.exists('./user_results/history'):
        os.makedirs('./user_results/history')


class ImageViewer:
    def __init__(self, parent, image_path):
        self.photo2 = None
        self.image2 = None
        self.root = tk.Toplevel()  # 使用 Toplevel 创建顶级窗口
        self.root.title("Image Viewer")  # 设置窗口标题
        self.root.geometry("900x600+320+150")
        self.parent = parent
        self.image_path = image_path
        if not os.path.exists('./color_map.png'):
            create_color_map_image()
        self.color_map = Image.open('./color_map.png')
        self.color_photo = ImageTk.PhotoImage(self.color_map)

        self.color_label = tk.Label(self.root, image=self.color_photo)
        self.color_label.pack(side='left', padx=0)

        # 创建顶部框架
        top_frame = tk.Frame(self.root)
        top_frame.pack(side='top', pady=0, fill='y', expand=True)

        # 创建底部框架
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side='bottom', pady=0, fill='y', expand=True)

        # 定义选项列表
        options = ["FCN8x", "UNet", "DeepLabV3"]

        # 创建变量存储选择结果
        self.selected_option = tk.StringVar(bottom_frame)
        self.selected_option.set(options[0])  # 设置默认选项

        # 创建下拉选择框
        option_menu = tk.OptionMenu(bottom_frame, self.selected_option, *options)
        option_menu.pack(side='left', padx=20)

        # 创建Button
        self.predict = tk.Button(bottom_frame, text="predict", command=self.predict, width=20, height=1,
                                 bg='#BDBDBD')
        self.predict.pack(side='left', padx=20)

        self.close = tk.Button(bottom_frame, text="close", command=self.close_window, width=20, height=1,
                               bg='#BDBDBD')
        self.close.pack(side='right', padx=20)

        # 打开并加载图片
        self.image = Image.open(image_path)
        self.image = self.image.resize((330, 330))
        # 创建 PhotoImage 对象
        self.photo = ImageTk.PhotoImage(self.image)

        # 创建 Label 组件来显示图片
        self.label = tk.Label(top_frame, image=self.photo)
        self.label.pack(side='left', padx=10)
        # 创建 Label2 组件来显示预测图片
        self.label2 = tk.Label(top_frame)
        self.label2.pack(side='right', padx=10)

    def close_window(self):
        self.root.destroy()
        self.parent.deiconify()

    def show(self):
        self.root.mainloop()  # 运行主循环，显示窗口

    def predict(self):
        model = self.selected_option.get()
        messagebox.showinfo("选择的选项", f"你选择了：{model}")
        pre_path = evaluate(self.image_path, model)
        # 打开并加载图片
        self.image2 = Image.open(pre_path)
        self.image2 = self.image2.resize((330, 330))
        # 创建 PhotoImage 对象
        self.photo2 = ImageTk.PhotoImage(self.image2)
        self.label2.config(image=self.photo2)


class StartWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Start Window")
        self.root.configure(bg='#BDBDBD')
        self.root.geometry("400x210+620+260")
        label_font = tkfont.Font(size=25, weight='bold', family='Tahoma')
        title_label = tk.Label(self.root, text="Welcome", font=label_font, fg='white', bg='#BDBDBD')
        title_label.pack(pady=15)

        # 创建底部框架
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side='bottom', pady=0)
        bottom_frame.configure(bg='#BDBDBD')
        # 创建顶部框架
        top_frame = tk.Frame(self.root)
        top_frame.pack(side='bottom', pady=0)
        top_frame.configure(bg='#BDBDBD')

        custom_font = tkfont.Font(size=9, weight='bold', family='Tahoma')
        his_button = tk.Button(bottom_frame, text='View History', command=self.view_his, width=20, height=2,
                               font=custom_font, bg='#9E9E9E', fg='white')
        start_button = tk.Button(top_frame, text="Select Picture", command=self.open_image, width=20, height=2,
                                 font=custom_font, bg='#9E9E9E', fg='white')

        start_button.pack(pady=10, side='left', padx=15)

        his_button.pack(pady=10, side='left', padx=15)

    def view_his(self):
        self.root.withdraw()
        image_his = ImageSwitcher(self.root)
        image_his.show()

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.root.withdraw()
            image_viewer = ImageViewer(image_path=file_path, parent=self.root)
            image_viewer.show()  # 显示 ImageViewer 窗口

    def show(self):
        self.root.mainloop()


if __name__ == "__main__":
    mk_dir()
    app = StartWindow()
    app.show()
