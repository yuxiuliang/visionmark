import requests

url = "http://localhost:5000/detect"
files = {'images': open('../images/3.png', 'rb')}  # 替换为你的测试图片路径 'classes': '2',
data = {"lang": "zh", "font_size": "3"}  # 语言、颜色、字体大小参数可选
response = requests.post(url, files=files, data=data)

# 保存结果
if response.status_code == 200:
    with open('test_annotated_result.jpg', 'wb') as f:
        f.write(response.content)
    print("检测完成，结果已保存为 annotated_result.jpg")
else:
    print(f"请求失败，状态码: {response.status_code}")
    print(f"错误信息: {response.text}")
