import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from clip_embedding import clip_embedding
from milvus_operator import text_image_vector
from net_helper import net_helper
import pandas as pd
from typing import List
from pathlib import Path
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化会话状态
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'search_history' not in st.session_state:
    st.session_state.search_history = []


def load_models():
    """加载模型并缓存到会话状态"""
    if not st.session_state.model_loaded:
        with st.spinner("正在加载模型..."):
            try:
                # 这里可以添加模型加载的代码
                # 如果模型已经通过 clip_embedding 模块加载，则不需要重复加载
                st.session_state.model_loaded = True
                logger.info("模型加载完成")
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")
                st.error(f"模型加载失败: {str(e)}")
                return False
    return True


def plot_images_with_scores(images: List[Image.Image], scores: List[float], rows: int = 1, cols: int = None):
    """
    展示多个图片并显示对应的分数

    参数:
    - images: PIL Image 对象列表
    - scores: 对应的分数列表
    - rows: 行数（默认为1）
    - cols: 列数（默认为None，将自动计算）
    """
    if cols is None:
        cols = len(images)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(images):
                axes[i, j].imshow(images[idx])
                axes[i, j].axis("off")
                axes[i, j].set_title(f"Score: {scores[idx]:.4f}", fontsize=20)

    plt.tight_layout()
    return fig


def image_search(text: str, num_results: int = 16):
    """执行图片搜索"""
    if not text:
        st.error("请输入搜索文本！")
        return None, None

    if not st.session_state.model_loaded:
        st.error("模型尚未加载完成，请稍候...")
        return None, None

    with st.spinner("正在搜索中..."):
        try:
            # CLIP编码
            input_embedding = clip_embedding.embedding_text(text)
            input_embedding = input_embedding[0].detach().cpu().numpy()

            # 搜索图片
            results = text_image_vector.search_data(input_embedding, limit=num_results)
            if not results:
                st.error("未找到匹配的图片！")
                return None, None

            # 准备数据
            pil_images = [Image.open(result['path']) for result in results]
            scores = [result.get('distance', 0) for result in results]

            # 创建结果DataFrame
            results_df = pd.DataFrame({
                '图片路径': [result['path'] for result in results],
                '相似度分数': scores
            })

            return pil_images, results_df
        except Exception as e:
            logger.error(f"搜索过程中发生错误: {str(e)}")
            st.error(f"搜索过程中发生错误: {str(e)}")
            return None, None


def main():
    try:
        st.set_page_config(
            page_icon="🔍",
            page_title="图文搜索系统",
            layout="wide",
        )

        # 加载模型
        if not load_models():
            st.error("模型加载失败，请检查日志")
            return

        # 侧边栏
        with st.sidebar:
            st.title("图文搜索")
            st.markdown("---")
            num_cols = st.number_input(
                label="显示列数",
                value=4,
                min_value=1,
                max_value=8,
                step=1
            )
            num_rows = st.number_input(
                label="显示行数",
                value=4,
                min_value=1,
                max_value=100,
                step=1
            )

        st.title("🔍 图文搜索系统")

        # 搜索输入
        search_text = st.text_input(
            "请输入搜索关键词：",
            placeholder="例如：一只可爱的猫咪",
            key="search_text"
        )

        if search_text:
            # 添加到搜索历史
            if search_text not in st.session_state.search_history:
                st.session_state.search_history.append(search_text)

            images, results_df = image_search(search_text, num_results=num_cols * num_rows)

            if images and results_df is not None:
                # 创建标签页
                tab1, tab2 = st.tabs(["图片展示", "搜索结果"])

                with tab1:
                    # 显示图片网格
                    fig = plot_images_with_scores(
                        images,
                        results_df['相似度分数'].tolist(),
                        rows=num_rows,
                        cols=num_cols
                    )
                    st.pyplot(fig)

                with tab2:
                    # 显示结果表格
                    st.dataframe(results_df)

        # 显示搜索历史
        if st.session_state.search_history:
            with st.expander("搜索历史"):
                for hist in reversed(st.session_state.search_history[-5:]):
                    st.text(hist)

    except Exception as e:
        logger.error(f"应用运行出错: {str(e)}")
        st.error(f"应用运行出错: {str(e)}")


if __name__ == "__main__":
    logger.info("启动图文搜索应用...")
    main()
    # 保持应用运行
    while True:
        time.sleep(1)
