import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw
import os

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("鼠标画板")
        
        # 设置画布
        self.canvas_width = 300
        self.canvas_height = 300
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()
        
        # 设置画笔
        self.pen_color = "black"
        self.pen_size = 5
        self.old_x = None
        self.old_y = None
        
        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        
        # 添加按钮
        button_frame = tk.Frame(root)
        button_frame.pack(fill=tk.X)
        
        clear_button = tk.Button(button_frame, text="清空画板", command=self.clear_canvas)
        clear_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        save_button = tk.Button(button_frame, text="保存图片", command=self.save_image)
        save_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        color_button = tk.Button(button_frame, text="选择颜色", command=self.choose_color)
        color_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        size_label = tk.Label(button_frame, text="画笔大小:")
        size_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.size_slider = tk.Scale(button_frame, from_=1, to=20, orient=tk.HORIZONTAL, command=self.change_size)
        self.size_slider.set(self.pen_size)
        self.size_slider.pack(side=tk.LEFT, padx=5, pady=5)
    
    def paint(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                   width=self.pen_size, fill=self.pen_color,
                                   capstyle=tk.ROUND, smooth=tk.TRUE)
        self.old_x = event.x
        self.old_y = event.y
    
    def reset(self, event):
        self.old_x = None
        self.old_y = None
    
    def clear_canvas(self):
        self.canvas.delete("all")
    
    def save_image(self):
        file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        initialfile="drawing.png"
    )
        if file_path:
            # 创建一个新的 Pillow 图像
            img = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
            draw = ImageDraw.Draw(img)
            
            # 获取画布上的所有项目并绘制到 Pillow 图像上
            for item in self.canvas.find_all():
                coords = self.canvas.coords(item)
                if self.canvas.type(item) == "line":
                    draw.line(coords, fill=self.pen_color, width=self.pen_size)
                # 可以添加对其他图形类型的支持
            
            img.save(file_path)
            print(f"图片已保存为 {file_path}")
    
    def choose_color(self):
        color = tk.colorchooser.askcolor(title="选择画笔颜色")[1]
        if color:
            self.pen_color = color
    
    def change_size(self, value):
        self.pen_size = int(value)


root = tk.Tk()
app = DrawingApp(root)
root.mainloop()
