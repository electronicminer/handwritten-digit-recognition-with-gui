import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw
import torch
import os
import tempfile
from digit_recognition import predict_image,detect_digits_line


class DigitDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别")
        
        # 设置主窗口大小和位置
        window_width = 500
        window_height = 600
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # 创建标题标签
        title_label = tk.Label(root, text="请在下方画板书写数字(0-9)", font=("Arial", 14))
        title_label.pack(pady=10)
        
        # 设置画布
        self.canvas_width = 380
        self.canvas_height = 380
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, 
                              bg="white", relief="ridge", bd=2)
        self.canvas.pack(pady=10)
        
        # 设置画笔
        self.pen_color = "black"
        self.pen_size =8
        self.old_x = None
        self.old_y = None
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.start_paint)  # 鼠标按下事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_and_predict)  # 合并重置和预测
        
        # 按钮框架
        button_frame = tk.Frame(root)
        button_frame.pack(fill=tk.X, padx=20)
        
        # 按钮
        clear_button = tk.Button(button_frame, text="清空画板", command=self.clear_canvas,
                               width=15, height=2)
        clear_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        # predict_button = tk.Button(button_frame, text="识别数字", command=self.predict_digit,
        #                          width=15, height=2)
        # predict_button.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # 预测结果标签
        self.result_label = tk.Label(root, text="识别结果：", font=("Arial", 24, "bold"))
        self.result_label.pack(pady=20)
    
    def start_paint(self, event):
        self.old_x = event.x
        self.old_y = event.y
    
    def paint(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                   width=self.pen_size, fill=self.pen_color,
                                   capstyle=tk.ROUND, smooth=tk.TRUE)
        self.old_x = event.x
        self.old_y = event.y
    
    def reset_and_predict(self, event):
        self.old_x = None
        self.old_y = None
        self.root.after(1000, self.predict_digit)

        # self.predict_digit()
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.result_label.config(text="识别结果：")
    
    def predict_digit(self):
        # 创建临时文件保存画布内容
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        
        # 创建一个新的 Pillow 图像
        img = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        draw = ImageDraw.Draw(img)
        
        # 获取画布上的所有项目并绘制到 Pillow 图像上
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if self.canvas.type(item) == "line":
                draw.line(coords, fill="black", width=self.pen_size)
        
        # 保存图像
        img.save(temp_file.name)
        
        # 调用预测函数
        try:
            prediction =detect_digits_line(temp_file.name)
            self.result_label.config(text=f"识别结果：{prediction}")
        except Exception as e:
            self.result_label.config(text=f"识别错误：{str(e)}")
        finally:
            # 删除临时文件
            temp_file.close()
            os.unlink(temp_file.name)

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitDrawingApp(root)
    root.mainloop()
