import json
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import jsonlines


class CaptionViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("图片描述查看器")

        # 设置窗口大小和位置
        window_width = 1200
        window_height = 800
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # 创建主框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建图片显示区域
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # 创建描述文本区域
        self.caption_frame = ttk.Frame(self.main_frame)
        self.caption_frame.pack(fill=tk.X, pady=10)

        self.caption_text = tk.Text(self.caption_frame, height=10, wrap=tk.WORD)
        self.caption_text.pack(fill=tk.X)

        # 创建控制区域
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=10)

        # 添加导航按钮
        self.prev_button = ttk.Button(self.control_frame, text="上一页", command=self.prev_page)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.page_label = ttk.Label(self.control_frame, text="0/0")
        self.page_label.pack(side=tk.LEFT, padx=5)

        self.next_button = ttk.Button(self.control_frame, text="下一页", command=self.next_page)
        self.next_button.pack(side=tk.LEFT, padx=5)

        # 添加跳转输入框
        ttk.Label(self.control_frame, text="跳转到:").pack(side=tk.LEFT, padx=5)
        self.page_entry = ttk.Entry(self.control_frame, width=10)
        self.page_entry.pack(side=tk.LEFT, padx=5)
        self.jump_button = ttk.Button(self.control_frame, text="跳转", command=self.jump_to_page)
        self.jump_button.pack(side=tk.LEFT, padx=5)

        # 初始化数据
        self.jsonl_path = "bdd100k_captions.jsonl"
        self.image_dir = r"E:\playground\ai\datasets\bdd100k\100K\bdd100k_images\bdd100k\images\100k\train"
        self.data = []
        self.current_page = 0
        self.load_data()

    def load_data(self):
        """加载JSONL文件数据"""
        try:
            with jsonlines.open(self.jsonl_path) as reader:
                self.data = list(reader)
            self.total_pages = len(self.data)
            self.update_page_label()
            self.show_current_page()
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")

    def show_current_page(self):
        """显示当前页的内容"""
        if not self.data:
            return

        current_item = self.data[self.current_page]
        image_id = current_item["imageId"]
        captions = current_item["short_caption_list"]

        # 显示图片
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        try:
            # 打开并调整图片大小
            image = Image.open(image_path)
            # 计算调整后的尺寸，保持宽高比
            window_width = self.image_frame.winfo_width()
            window_height = self.image_frame.winfo_height()
            if window_width > 1 and window_height > 1:  # 确保窗口已经创建
                image.thumbnail((window_width, window_height))
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # 保持引用
        except Exception as e:
            print(f"加载图片时出错: {str(e)}")
            self.image_label.configure(image='')

        # 显示描述
        self.caption_text.delete(1.0, tk.END)
        self.caption_text.insert(tk.END, f"图片ID: {image_id}\n\n")
        for i, caption in enumerate(captions, 1):
            self.caption_text.insert(tk.END, f"{i}. {caption}\n")

    def update_page_label(self):
        """更新页码标签"""
        self.page_label.configure(text=f"{self.current_page + 1}/{self.total_pages}")

    def prev_page(self):
        """显示上一页"""
        if self.current_page > 0:
            self.current_page -= 1
            self.update_page_label()
            self.show_current_page()

    def next_page(self):
        """显示下一页"""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.update_page_label()
            self.show_current_page()

    def jump_to_page(self):
        """跳转到指定页"""
        try:
            page = int(self.page_entry.get()) - 1
            if 0 <= page < self.total_pages:
                self.current_page = page
                self.update_page_label()
                self.show_current_page()
            else:
                print(f"页码超出范围: 1-{self.total_pages}")
        except ValueError:
            print("请输入有效的页码")


def main():
    root = tk.Tk()
    app = CaptionViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
