1. 安装 Python 3.6 https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
2. ctrl + r -> cmd 进命令行，输入 pip install opencv-python
3. 运行目录下的 run.bat

run.bat 中的内容解释
python video2image.py --source 0 --w 1024 --h 768 --shotdir img

--source 0 摄像头设备的编号，从零开始
--w 1024 画面宽度，不需要修改
--h 768 画面高度，不需要修改
--shotdir img 输出的文件夹名字，建议一只鞋子配一个文件夹