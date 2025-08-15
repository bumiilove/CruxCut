 <!--# 입출력 영상-->
<p align="center">
    <img width=70% src=https://github.com/user-attachments/assets/04c2a39e-0c58-4f15-a95e-ad9d91af3f3c>
</p>

# 🧗 Just Climb. Your highlight is ready.
이제 클라이밍 영상을 직접 하나하나 편집하느라 애쓰지 않아도 됩니다. 

<!--# 랜딩 페이지 배너-->
<p align="center">
    <img width=80% src="https://github.com/user-attachments/assets/e36fdcb9-86cf-4aa9-a292-ab1581c7294f" />
<br><br/>
</p>

<!--# 뱃지-->
<div align="center">
<img src=https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54>
<img src=https://img.shields.io/badge/vercel-%23000000.svg?style=for-the-badge&logo=vercel&logoColor=white>
<img src=https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white>
</div>

### 바로 경험해보세요. [👉 CruxCut](https://crux-cut.vercel.app)

# CruxCut
CruxCut AI는 등반 영상에서 여러분들에게 포커스된 영상을 자동으로 생성해주는 클라이밍 영상 자동 편집 AI 서비스 입니다.

## Introduction
 CruxCut AI는 클라이밍 영상 편집 시간을 53% 단축할 뿐만 아니라, 심지어 수동으로 영상을 편집할 필요도 없습니다. 여러분은 그저 벽을 오르고, 편집은 저희 CruxCut AI에게 맡기세요.

 최근 실내 볼더링의 인기 증가로 클라이밍 영상을 촬영해 SNS에 공유하는 문화가 확산되었습니다. 삼각대를 놓고 클라이밍 영상을 촬영하다보면, 종종 다른 인물이 등장하거나 멋진 무브를 더 다이나믹하게 담아내지 못해 아쉬울 때가 있습니다. 물론, 좋은 영상 편집 프로그램들 덕분에 고정된 카메라로도 나를 따라 줌인된 클라이밍 영상을 직접 편집할 수 있습니다.

 저희 CruxCut은 AI 기술로, 번거롭고 어려운 수동 편집 과정 없이도 여러분을 따라다니는 클라이밍 영상을 자동으로 생성하도록 돕고자 개발되었습니다. CruxCut은 1분 이내의 클라이밍 영상을 업로드하기만 하면, AI 기반 등반자 검출 알고리즘과 등반자 추적을 통해 여러분을 중심으로 클라이밍 영상을 재구성해드리는 AI 서비스입니다. 

 저희 서비스는 클라이밍을 시작으로, 축구, 야구 등 원거리에서 촬영 후 송출되는 스포츠와 K-POP, 뮤지컬 등 공연 예술 영상을 자동 편집하는 통합 솔루션 개발을 목표로 향후 다양한 도메인에 기능을 적용, 확장해나갈 것입니다.
 
## 주요 기능

-   **AI 기반 사용자 검출**: 고급, 경량 객체 검출 모델([YOLOv11](https://github.com/ultralytics/ultralytics))을 활용하여 영상 내 클라이머를 정확하게 인식하고 추적합니다.
-   **포커싱된 영상 생성**: 사용자가 항상 화면 중앙에 오도록 영상을 크롭하거나 이동시켜 몰입도를 높입니다.
-   **간편한 업로드/다운로드**: 원본 영상 업로드 및 편집된 영상 다운로드 기능을 직관적인 UI로 제공합니다.
-   **다양한 영상 포맷 지원**: MP4, AVI 등 주요 영상 포맷을 지원하여 사용성을 높입니다.

<p align="center">
:exclamation: CrxuCut AI가 클라이밍 영상에서 어떻게 여러분을 포착하는지 직접 확인하세요 :exclamation:
</p>

[![CruxCut tracks your move](https://img.youtube.com/vi/FbD5ZKpMjNA/maxresdefault.jpg)](https://www.youtube.com/shorts/FbD5ZKpMjNA) 

 # Demo
CruxCut은 업로드, 다운로드 두 기능만으로 효율적인 영상 자동 편집 기능을 제공합니다. 클라이밍 중 수시로 영상을 업로드하고 편집해보세요.
<p align="center">
<img width=30% src=https://github.com/user-attachments/assets/888a534d-6092-47c1-b20e-dae141e58009>
</p>

## Quick guide

### build
시작하기전 아래의 명령어를 통해 파이썬 패키지를 workplace에 설치해주세요.
```shell
pip install -r requirements.txt
```
### run
backend api 호출은 아래의 명령어를 통해 실행됩니다.
```shell
uvicorn main:app --reload --host 0.0.0.0 --port your_port
```
영상편집은 아래의 명령어를 통해 실행됩니다.
```shell
python scripts/video_cropper_oop.py --input [입력 비디오 경로] --output [출력 저장 경로]
```


## License

## FAQ

## Contact us
<table  style="border-collapse: collapse; border: none;">
  <tr>
          <td align="center" width="250px" style="padding: 0 20px; border: none;">
          <img src="https://github.com/user-attachments/assets/d5afbc9a-0ab2-4428-8865-f1efbc5d27ec" width="120px" style="border-radius: 50%;" alt=""/>
          <br /><b>권경범</b>
          <br /><a href="mailto:bumiilove@gmail.com">bumiilove@gmail.com</a>
    <!--       <br /><sub>소속1</sub> -->
    </td>
         <td align="center" width="250px" style="padding: 0 20px; border: none;">
         <img src="https://github.com/user-attachments/assets/c0b769b0-2040-4fbb-890e-f9ea51aa1db7" width="120px" style="border-radius: 50%;" alt=""/>
         <br /><b>정성훈</b>
         <br /><a href="mailto:hoon0867@gmail.com">hoon0867@gmail.com</a>
   <!--       <br /><sub>소속2</sub> -->
    </td>
        <td align="center" width="250px" style="padding: 0 20px; border: none;">
        <img src="https://github.com/user-attachments/assets/06c5fa42-c487-4f22-b126-b730a746b7ac" width="120px" style="border-radius: 50%;" alt=""/>          
        <br /><b>민 건</b>
        <br /><a href="mgun1520@gmail.com">mgun1520@gmail.com</a>
  <!--       <br /><sub>소속3</sub> -->
    </td>
  </tr>
</table>
