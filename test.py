import tkinter as tk


def print_selected_option():
    selected_value = selected_option.get()
    print(f"Selected option: {selected_value}")


root = tk.Tk()
root.title("OptionMenu 示例")

bottom_frame = tk.Frame(root)
bottom_frame.pack(pady=20)

options = ["选项1", "选项2", "选项3", "选项4"]

selected_option = tk.StringVar(bottom_frame)
selected_option.set(options[0])  # 设置默认选项

option_menu = tk.OptionMenu(bottom_frame, selected_option, *options)
option_menu.pack(side='left')

button = tk.Button(bottom_frame, text="Print Selected Option", command=print_selected_option)
button.pack(side='left', padx=10)

root.mainloop()
