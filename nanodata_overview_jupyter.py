"""
纳米数据概览图生成器 - Jupyter优化版
支持在Jupyter Notebook中直接调用，提供更好的交互体验
"""

import os
import sys
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import matplotlib
matplotlib.use('Agg')  # 不使用GUI
import matplotlib.pyplot as plt
from datetime import datetime
import traceback
import warnings
import json
from IPython.display import display, Image, HTML, clear_output
import ipywidgets as widgets
from IPython.core.interactiveshell import InteractiveShell

warnings.filterwarnings('ignore', category=UserWarning)

# 尝试导入必要的库
try:
    import nanonispy2 as nap
    NANONISPY_AVAILABLE = True
except ImportError:
    NANONISPY_AVAILABLE = False
    print("注意: 未找到nanonispy库，无法处理.sxm文件")
    print("安装: pip install nanonispy")

try:
    import gwyfile
    GWYFILE_AVAILABLE = True
except ImportError:
    GWYFILE_AVAILABLE = False
    print("注意: 未找到gwyfile库，无法处理.gwy文件")
    print("安装: pip install gwyfile")

class JupyterNanoDataOverviewGenerator:
    """Jupyter环境下的纳米数据概览图生成器"""
    
    def __init__(self):
        """初始化Jupyter生成器"""
        self.progress_bar = None
        self.status_label = None
        self.current_file_label = None
        self.output_area = None
        self.results = {}
        self.is_jupyter = self._check_jupyter_environment()
        
    def _check_jupyter_environment(self) -> bool:
        """检查是否在Jupyter环境中运行"""
        try:
            from IPython import get_ipython
            ip = get_ipython()
            return ip is not None and 'IPKernelApp' in ip.config
        except:
            return False
    
    def create_widgets(self) -> Dict[str, widgets.Widget]:
        """创建交互式控件"""
        # 目录选择控件
        dir_input = widgets.Text(
            value='',
            placeholder='输入目录路径 (如: /path/to/data)',
            description='目录:',
            layout=widgets.Layout(width='80%')
        )
        
        # 文件类型选择
        file_type_dropdown = widgets.Dropdown(
            options=['.sxm', '.gwy'],
            value='.sxm',
            description='文件类型:',
            disabled=not (NANONISPY_AVAILABLE or GWYFILE_AVAILABLE)
        )
        
        # 每行图片数量
        cols_slider = widgets.IntSlider(
            value=3,
            min=1,
            max=6,
            step=1,
            description='每行图片:',
            continuous_update=False
        )
        
        # 每图最大图片数
        max_per_fig_slider = widgets.IntSlider(
            value=12,
            min=1,
            max=30,
            step=1,
            description='每图最多:',
            continuous_update=False
        )
        
        # 图片尺寸
        width_slider = widgets.IntSlider(
            value=4,
            min=2,
            max=8,
            step=1,
            description='宽度(英寸):',
            continuous_update=False
        )
        
        height_slider = widgets.IntSlider(
            value=3,
            min=2,
            max=8,
            step=1,
            description='高度(英寸):',
            continuous_update=False
        )
        
        # 处理按钮
        process_button = widgets.Button(
            description='开始处理',
            button_style='primary',
            layout=widgets.Layout(width='150px', height='40px')
        )
        
        # 进度显示
        self.progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            description='进度:',
            bar_style='info',
            layout=widgets.Layout(width='80%')
        )
        
        self.status_label = widgets.Label(value='准备就绪')
        self.current_file_label = widgets.Label(value='')
        
        # 结果展示区域
        self.output_area = widgets.Output()
        
        return {
            'dir_input': dir_input,
            'file_type': file_type_dropdown,
            'cols': cols_slider,
            'max_per_fig': max_per_fig_slider,
            'width': width_slider,
            'height': height_slider,
            'process_button': process_button,
            'progress_bar': self.progress_bar,
            'status_label': self.status_label,
            'current_file_label': self.current_file_label,
            'output_area': self.output_area
        }
    
    def display_ui(self):
        """显示用户界面"""
        if not self.is_jupyter:
            print("此功能仅在Jupyter环境中可用")
            return
        
        widgets_dict = self.create_widgets()
        
        # 控件布局
        controls = widgets.VBox([
            widgets.HBox([widgets_dict['dir_input']]),
            widgets.HBox([widgets_dict['file_type'], widgets_dict['cols'], widgets_dict['max_per_fig']]),
            widgets.HBox([widgets_dict['width'], widgets_dict['height']]),
            widgets.HBox([widgets_dict['process_button']]),
            widgets.HBox([widgets_dict['progress_bar']]),
            widgets.HBox([widgets_dict['status_label']]),
            widgets.HBox([widgets_dict['current_file_label']])
        ])
        
        # 显示所有控件
        display(controls)
        display(widgets_dict['output_area'])
        
        # 绑定按钮事件
        widgets_dict['process_button'].on_click(
            lambda b: self.process_with_widgets(
                widgets_dict['dir_input'].value,
                widgets_dict['file_type'].value,
                widgets_dict['cols'].value,
                widgets_dict['max_per_fig'].value,
                widgets_dict['width'].value,
                widgets_dict['height'].value
            )
        )
    
    def process_with_widgets(self, directory: str, file_type: str, cols_per_row: int,
                            max_per_fig: int, width: int, height: int):
        """通过控件处理数据"""
        with self.output_area:
            clear_output()
            self.generate_overview(directory, file_type, cols_per_row, max_per_fig, width, height)
    
    def update_progress(self, value: int, message: str = ""):
        """更新进度条"""
        if self.progress_bar:
            self.progress_bar.value = value
        if self.status_label and message:
            self.status_label.value = message
    
    def update_current_file(self, filename: str):
        """更新当前处理文件显示"""
        if self.current_file_label:
            self.current_file_label.value = f"当前文件: {filename}"
    
    def generate_overview(self, base_dir: str, file_type: str, cols_per_row: int = 3,
                         max_per_fig: int = 12, width: int = 4, height: int = 3):
        """生成概览图（可在Jupyter中直接调用）"""
        
        # 验证输入
        if not base_dir or not os.path.exists(base_dir):
            print(f"错误: 目录不存在: {base_dir}")
            return
        
        if file_type not in ['.sxm', '.gwy']:
            print("错误: 文件类型必须是 '.sxm' 或 '.gwy'")
            return
        
        if file_type == '.sxm' and not NANONISPY_AVAILABLE:
            print("错误: 需要安装nanonispy库来处理.sxm文件")
            print("请运行: pip install nanonispy")
            return
        
        if file_type == '.gwy' and not GWYFILE_AVAILABLE:
            print("错误: 需要安装gwyfile库来处理.gwy文件")
            print("请运行: pip install gwyfile")
            return
        
        print("=" * 60)
        print("纳米数据概览图生成器 - Jupyter版")
        print("=" * 60)
        print(f"基础目录: {base_dir}")
        print(f"文件类型: {file_type}")
        print(f"每行图片数: {cols_per_row}")
        print(f"每图最多图片数: {max_per_fig}")
        print("=" * 60)
        
        # 递归处理目录
        self._process_directory_recursive(
            Path(base_dir), 
            file_type, 
            cols_per_row, 
            max_per_fig, 
            (width, height)
        )
        
        print("=" * 60)
        print("处理完成!")
        print("=" * 60)
        
        # 显示统计信息
        self._display_summary()
    
    def _process_directory_recursive(self, current_dir: Path, file_type: str,
                                     cols_per_row: int, max_per_fig: int,
                                     figsize_per_img: Tuple[int, int]):
        """递归处理目录"""
        # 处理当前目录
        self._process_single_directory(current_dir, file_type, cols_per_row, max_per_fig, figsize_per_img)
        
        # 递归处理子目录
        for item in current_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                self._process_directory_recursive(item, file_type, cols_per_row, max_per_fig, figsize_per_img)
    
    def _process_single_directory(self, current_dir: Path, file_type: str,
                                 cols_per_row: int, max_per_fig: int,
                                 figsize_per_img: Tuple[int, int]):
        """处理单个目录"""
        # 查找文件
        file_pattern = f"*{file_type}"
        files = list(current_dir.glob(file_pattern))
        
        if not files:
            return
        
        print(f"\n处理目录: {current_dir}")
        print(f"找到 {len(files)} 个 {file_type} 文件")
        
        # 读取所有文件数据
        image_data_list = []
        file_names = []
        failed_files = []
        
        for i, file_path in enumerate(files):
            self.update_progress(int((i / len(files)) * 50), f"读取文件: {i+1}/{len(files)}")
            self.update_current_file(file_path.name)
            
            try:
                if file_type == '.sxm':
                    data = self._read_sxm_file(file_path)
                else:  # .gwy
                    data = self._read_gwy_file(file_path)
                
                if data is not None:
                    image_data_list.append(data)
                    file_names.append(file_path.name)
                else:
                    failed_files.append(file_path.name)
            except Exception as e:
                print(f"读取文件 {file_path.name} 时出错: {str(e)[:100]}")
                failed_files.append(file_path.name)
        
        if not image_data_list:
            print("没有成功读取任何文件")
            return
        
        if failed_files:
            print(f"以下文件读取失败: {', '.join(failed_files[:5])}")
            if len(failed_files) > 5:
                print(f"... 以及另外 {len(failed_files) - 5} 个文件")
        
        # 生成概览图
        self.update_progress(60, "生成概览图...")
        self._generate_overview_images(image_data_list, file_names, current_dir, 
                                      cols_per_row, max_per_fig, figsize_per_img)
        
        # 记录结果
        self.results[str(current_dir)] = {
            'total_files': len(files),
            'successful_files': len(image_data_list),
            'failed_files': len(failed_files),
            'overview_generated': True
        }
    
    def _read_sxm_file(self, file_path: Path) -> Optional[np.ndarray]:
        """读取.sxm文件"""
        try:
            scan = nap.read.Scan(str(file_path))
            
            # 尝试获取数据
            channels = scan._load_data()
            channel = channels['Z']
            data = channel['forward']
            return data
            
        except Exception as e:
            print(f"读取.sxm文件 {file_path.name} 时出错: {str(e)[:100]}")
            return None
    
    def _read_gwy_file(self, file_path: Path) -> Optional[np.ndarray]:
        """读取.gwy文件"""
        try:
            container = gwyfile.load(str(file_path))  # 加载.gwy文件
            data_fields = gwyfile.util.get_datafields(container)  # 获取所有数据通道
            if not data_fields:
                raise ValueError("No data fields found in the file")
            data_field = data_fields['Z (Forward)']  # 取第一个数据通道
            data = data_field.data  # 获取数据数组
            return data

        except Exception as e:
            print(f"读取.gwy文件 {file_path.name} 时出错: {str(e)[:100]}")
            return None
    
    def _generate_overview_images(self, image_data_list: List[np.ndarray], 
                                 file_names: List[str], save_dir: Path,
                                 cols_per_row: int, max_per_fig: int,
                                 figsize_per_img: Tuple[int, int]):
        """生成概览图"""
        num_images = len(image_data_list)
        num_figures = math.ceil(num_images / max_per_fig)
        
        generated_files = []
        
        for fig_idx in range(num_figures):
            # 更新进度
            progress = 60 + int((fig_idx / num_figures) * 40)
            self.update_progress(progress, f"生成概览图 {fig_idx+1}/{num_figures}")
            
            # 获取当前批次的图片
            start_idx = fig_idx * max_per_fig
            end_idx = min(start_idx + max_per_fig, num_images)
            current_images = image_data_list[start_idx:end_idx]
            current_names = file_names[start_idx:end_idx]
            
            # 计算布局
            num_current = len(current_images)
            rows = math.ceil(num_current / cols_per_row)
            cols = min(cols_per_row, num_current)
            
            # 创建图形
            fig_width = cols * figsize_per_img[0]
            fig_height = rows * figsize_per_img[1]
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=150)
            
            # 添加子图
            for i, (img_data, filename) in enumerate(zip(current_images, current_names)):
                ax = plt.subplot(rows, cols, i + 1)
                
                if img_data is not None and len(img_data.shape) == 2:
                    im = ax.imshow(img_data, cmap='viridis', aspect='auto', 
                                  interpolation='nearest', origin='lower')
                    
                    # 只在第一个子图添加颜色条
                    if i == 0:
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # 设置标题
                short_name = filename[:20] + '...' if len(filename) > 20 else filename
                ax.set_title(short_name, fontsize=8, pad=2)
                ax.axis('off')
            
            # 添加大标题
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if num_figures > 1:
                fig.suptitle(f'Nano Data Overview - {save_dir.name} ({fig_idx+1}/{num_figures})\n'
                           f'Generated: {timestamp}', fontsize=12, y=0.98)
            else:
                fig.suptitle(f'Nano Data Overview - {save_dir.name}\n'
                           f'Generated: {timestamp}', fontsize=12, y=0.98)
            
            plt.tight_layout(rect=[0, 0.02, 1, 0.95])
            
            # 保存图片
            if num_figures > 1:
                save_name = f'overview_{save_dir.name}_{fig_idx+1:02d}.png'
            else:
                save_name = f'overview_{save_dir.name}.png'
            
            save_path = save_dir / save_name
            plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
            generated_files.append(str(save_path))
            '''
            # 在Jupyter中显示生成的图片
            if self.is_jupyter:
                with self.output_area:
                    display(Image(filename=str(save_path)))
                    print(f"已保存概览图: {save_path}")
            
            plt.close(fig)
            '''
        self.update_progress(100, "处理完成")
        return generated_files
    
    def _display_summary(self):
        """显示处理摘要"""
        if not self.results:
            print("没有生成任何概览图")
            return
        
        print("\n" + "=" * 60)
        print("处理摘要:")
        print("=" * 60)
        
        total_dirs = len(self.results)
        total_files = sum(r['total_files'] for r in self.results.values())
        successful_files = sum(r['successful_files'] for r in self.results.values())
        failed_files = sum(r['failed_files'] for r in self.results.values())
        
        print(f"处理目录数: {total_dirs}")
        print(f"总文件数: {total_files}")
        print(f"成功读取文件: {successful_files}")
        print(f"读取失败文件: {failed_files}")
        print(f"成功率: {successful_files/total_files*100:.1f}%" if total_files > 0 else "成功率: N/A")
        
        if self.results:
            print("\n已处理的目录:")
            for dir_path, stats in self.results.items():
                print(f"  {dir_path}: {stats['successful_files']}/{stats['total_files']} 个文件")
        
        # 在Jupyter中显示HTML摘要
        if self.is_jupyter:
            html_content = f"""
            <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <h4>处理摘要</h4>
                <p>处理目录数: {total_dirs}</p>
                <p>总文件数: {total_files}</p>
                <p>成功读取文件: {successful_files}</p>
                <p>读取失败文件: {failed_files}</p>
                <p>成功率: {successful_files/total_files*100:.1f}%</p>
            </div>
            """
            display(HTML(html_content))

class NanoDataOverviewGenerator:
    """命令行版本的纳米数据概览图生成器"""
    
    def __init__(self, base_dir: str, file_type: str, cols_per_row: int = 3, 
                 max_per_fig: int = 12, figsize_per_img: Tuple[int, int] = (4, 3)):
        """
        初始化生成器
        
        参数:
            base_dir: 基础目录路径
            file_type: 文件类型，'.sxm' 或 '.gwy'
            cols_per_row: 每行最多图片数量
            max_per_fig: 每张概览图最多图片数量
            figsize_per_img: 每个子图的尺寸(宽, 高)
        """
        self.base_dir = Path(base_dir)
        self.file_type = file_type.lower()
        self.cols_per_row = cols_per_row
        self.max_per_fig = max_per_fig
        self.figsize_per_img = figsize_per_img
        
        if self.file_type not in ['.sxm', '.gwy']:
            raise ValueError("文件类型必须是 '.sxm' 或 '.gwy'")
            
        if self.file_type == '.sxm' and not NANONISPY_AVAILABLE:
            raise ImportError("需要nanonispy库处理.sxm文件")
            
        if self.file_type == '.gwy' and not GWYFILE_AVAILABLE:
            raise ImportError("需要gwyfile库处理.gwy文件")
    
    def process_directory(self, current_dir: Path) -> None:
        """处理单个目录"""
        # 查找指定类型的文件
        file_pattern = f"*{self.file_type}"
        files = list(current_dir.glob(file_pattern))
        
        if not files:
            return
        
        print(f"处理目录: {current_dir}")
        print(f"找到 {len(files)} 个 {self.file_type} 文件")
        
        # 读取所有文件数据
        image_data_list = []
        file_names = []
        failed_files = []
        
        for file_path in files:
            try:
                if self.file_type == '.sxm':
                    data = self._read_sxm_file(file_path)
                else:  # .gwy
                    data = self._read_gwy_file(file_path)
                
                if data is not None:
                    image_data_list.append(data)
                    file_names.append(file_path.name)
                else:
                    failed_files.append(file_path.name)
            except Exception as e:
                print(f"读取文件 {file_path.name} 时出错: {str(e)[:100]}")
                failed_files.append(file_path.name)
        
        if not image_data_list:
            print("没有成功读取任何文件")
            return
        
        if failed_files:
            print(f"以下文件读取失败: {', '.join(failed_files[:5])}")
            if len(failed_files) > 5:
                print(f"... 以及另外 {len(failed_files) - 5} 个文件")
        
        # 生成概览图
        self._generate_overview_images(image_data_list, file_names, current_dir)
    
    def _read_sxm_file(self, file_path: Path) -> Optional[np.ndarray]:
        """读取.sxm文件"""
        try:
            data = nap.read(file_path)
            
            if hasattr(data, 'signals'):
                for signal_name in data.signals.keys():
                    signal_data = data.signals[signal_name]
                    if isinstance(signal_data, np.ndarray) and len(signal_data.shape) == 2:
                        return signal_data
                
                for attr in ['Z', 'Current', 'Topography', 'data']:
                    if hasattr(data, attr):
                        channel_data = getattr(data, attr)
                        if isinstance(channel_data, np.ndarray) and len(channel_data.shape) == 2:
                            return channel_data
            
            return None
            
        except Exception as e:
            print(f"读取.sxm文件 {file_path.name} 时出错: {str(e)[:100]}")
            return None
    
    def _read_gwy_file(self, file_path: Path) -> Optional[np.ndarray]:
        """读取.gwy文件"""
        try:
            gwy_obj = gwyfile.load(str(file_path))
            
            for key, value in gwy_obj.items():
                if isinstance(value, dict) and 'data' in value:
                    data_array = value['data']
                    if isinstance(data_array, np.ndarray) and len(data_array.shape) == 2:
                        return data_array
            
            return None
            
        except Exception as e:
            print(f"读取.gwy文件 {file_path.name} 时出错: {str(e)[:100]}")
            return None
    
    def _generate_overview_images(self, image_data_list: List[np.ndarray], 
                                 file_names: List[str], save_dir: Path) -> None:
        """生成概览图"""
        num_images = len(image_data_list)
        num_figures = math.ceil(num_images / self.max_per_fig)
        
        for fig_idx in range(num_figures):
            start_idx = fig_idx * self.max_per_fig
            end_idx = min(start_idx + self.max_per_fig, num_images)
            current_images = image_data_list[start_idx:end_idx]
            current_names = file_names[start_idx:end_idx]
            
            num_current = len(current_images)
            rows = math.ceil(num_current / self.cols_per_row)
            cols = min(self.cols_per_row, num_current)
            
            fig_width = cols * self.figsize_per_img[0]
            fig_height = rows * self.figsize_per_img[1]
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=150)
            
            for i, (img_data, filename) in enumerate(zip(current_images, current_names)):
                ax = plt.subplot(rows, cols, i + 1)
                
                if img_data is not None and len(img_data.shape) == 2:
                    im = ax.imshow(img_data, cmap='viridis', aspect='auto', 
                                  interpolation='nearest', origin='lower')
                    
                    if i == 0:
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                short_name = filename[:20] + '...' if len(filename) > 20 else filename
                ax.set_title(short_name, fontsize=8, pad=2)
                ax.axis('off')
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if num_figures > 1:
                fig.suptitle(f'Nano Data Overview - {save_dir.name} ({fig_idx+1}/{num_figures})\n'
                           f'Generated: {timestamp}', fontsize=12, y=0.98)
            else:
                fig.suptitle(f'Nano Data Overview - {save_dir.name}\n'
                           f'Generated: {timestamp}', fontsize=12, y=0.98)
            
            plt.tight_layout(rect=[0, 0.02, 1, 0.95])
            
            if num_figures > 1:
                save_name = f'overview_{save_dir.name}_{fig_idx+1:02d}.png'
            else:
                save_name = f'overview_{save_dir.name}.png'
            
            save_path = save_dir / save_name
            plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
            print(f"已保存概览图: {save_path}")
            
            plt.close(fig)
    
    def recursive_process(self, current_dir: Optional[Path] = None) -> None:
        """递归处理所有目录"""
        if current_dir is None:
            current_dir = self.base_dir
        
        self.process_directory(current_dir)
        
        for item in current_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                self.recursive_process(item)
    
    def run(self) -> None:
        """运行处理流程"""
        print("=" * 60)
        print("纳米数据概览图生成器")
        print("=" * 60)
        print(f"基础目录: {self.base_dir}")
        print(f"文件类型: {self.file_type}")
        print(f"每行图片数: {self.cols_per_row}")
        print(f"每图最多图片数: {self.max_per_fig}")
        print("=" * 60)
        
        if not self.base_dir.exists():
            print(f"错误: 目录不存在: {self.base_dir}")
            return
        
        try:
            self.recursive_process()
            print("=" * 60)
            print("处理完成!")
            print("=" * 60)
        except Exception as e:
            print(f"处理过程中出错: {str(e)}")
            traceback.print_exc()


def main():
    """主函数：命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='生成纳米数据文件(.sxm/.gwy)的拼接概览图，递归处理目录',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python nanodata_overview_jupyter.py /path/to/data -t .sxm -c 4 -m 16
  python nanodata_overview_jupyter.py /path/to/data -t .gwy
  
Jupyter使用:
  generator = JupyterNanoDataOverviewGenerator()
  generator.generate_overview('/path/to/data', '.sxm', cols_per_row=3, max_per_fig=12)
        """
    )
    
    parser.add_argument('directory', nargs='?', help='要处理的目录路径')
    parser.add_argument('-t', '--type', choices=['.sxm', '.gwy'], 
                       help='要处理的文件类型')
    parser.add_argument('-c', '--cols', type=int, default=3,
                       help='每行最多图片数量 (默认: 3)')
    parser.add_argument('-m', '--max', type=int, default=12,
                       help='每张概览图最多图片数量 (默认: 12)')
    parser.add_argument('--width', type=int, default=4,
                       help='每个子图的宽度(英寸) (默认: 4)')
    parser.add_argument('--height', type=int, default=3,
                       help='每个子图的高度(英寸) (默认: 3)')
    
    args = parser.parse_args()
    
    # 检查是否在Jupyter中运行
    try:
        from IPython import get_ipython
        ip = get_ipython()
        in_jupyter = ip is not None and 'IPKernelApp' in ip.config
    except:
        in_jupyter = False
    
    if in_jupyter and (args.directory is None or args.type is None):
        print("检测到在Jupyter环境中运行")
        print("使用以下方式之一:")
        print("1. 使用交互式界面:")
        print("   generator = JupyterNanoDataOverviewGenerator()")
        print("   generator.display_ui()")
        print("")
        print("2. 直接调用:")
        print("   generator.generate_overview('/path/to/data', '.sxm', cols_per_row=3)")
        return
    
    if args.directory is None or args.type is None:
        parser.print_help()
        sys.exit(1)
    
    # 检查必要的库是否安装
    if args.type == '.sxm' and not NANONISPY_AVAILABLE:
        print("错误: 需要安装nanonispy库来处理.sxm文件")
        print("请运行: pip install nanonispy")
        sys.exit(1)
    
    if args.type == '.gwy' and not GWYFILE_AVAILABLE:
        print("错误: 需要安装gwyfile库来处理.gwy文件")
        print("请运行: pip install gwyfile")
        sys.exit(1)
    
    # 创建并运行生成器
    try:
        generator = NanoDataOverviewGenerator(
            base_dir=args.directory,
            file_type=args.type,
            cols_per_row=args.cols,
            max_per_fig=args.max,
            figsize_per_img=(args.width, args.height)
        )
        generator.run()
    except ValueError as e:
        print(f"参数错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"运行错误: {e}")
        traceback.print_exc()
        sys.exit(1)


# Jupyter使用示例
def jupyter_usage_example():
    """显示Jupyter使用示例"""
    example_code = '''
# 在Jupyter中使用交互式界面
generator = JupyterNanoDataOverviewGenerator()
generator.display_ui()

# 或者在Jupyter中直接调用
# generator.generate_overview('/path/to/data', '.sxm', cols_per_row=3, max_per_fig=12)
'''
    print("Jupyter使用示例:")
    print(example_code)


if __name__ == "__main__":
    # 检查是否是直接从Python运行（而不是导入）
    if len(sys.argv) > 1 and sys.argv[0].endswith('.py'):
        main()
    else:
        # 如果在交互式环境中，显示帮助信息
        print("纳米数据概览图生成器")
        print("=" * 50)
        print("命令行使用: python nanodata_overview_jupyter.py /path/to/data -t .sxm")
        print("")
        print("Jupyter使用:")
        print("generator = JupyterNanoDataOverviewGenerator()")
        print("generator.display_ui()  # 显示交互式界面")
        print("")
        print("或者直接调用:")
        print("generator.generate_overview('/path/to/data', '.sxm')")
