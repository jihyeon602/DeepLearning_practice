import requests
import time


def predict_image(image_path):
    print(image_path)
    url = f"http://{"your_ip_port"}/?image={image_path}"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(data)
    else:
        print("Error:", response.json())


s_time = time.time()
for i in range(100):
    predict_image('./DL6/q5/img/seven.png') #이미지 경로
e_time = time.time()

print('걸린시간: ', e_time -  s_time)