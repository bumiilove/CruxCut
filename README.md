<!--# 입출력 영상-->
<p align="center">
<img width=70% src=https://github.com/user-attachments/assets/04c2a39e-0c58-4f15-a95e-ad9d91af3f3c>
</p>

# 🧗 Just Climb. Your highlight is ready.
이제 1분짜리 클라이밍 영상을 직접 하나하나 편집하느라 애쓰지 않아도 됩니다. 

~~클릭 한 번으로 수동 편집을 자동화 – 긴 편집 과정을 즉석에서 단축하는 AI 클라이밍 영상 편집~~ 

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

# Introduction
단락 1) Motivation - 
단락 2) 우리의 방법 간단 소개
클라이밍 영상은 대부분 고정된 카메라로 촬영되며, 촬영된 전체 영상에서 실제 등반 장면이나 하이라이트 순간만을 추출해 편집하는 과정은 시간과 노력이 많이 소모되는 반복 작업입니다. 특히, 등반자가 프레임 내에서 작아지거나 움직임이 많을 경우, 이를 수작업으로 식별하고 잘라내는 것은 더욱 번거롭고 비효율적입니다. 이러한 문제는 개인 크리에이터뿐만 아니라 스포츠 미디어 제작자에게도 생산성 저하와 콘텐츠 품질 저하로 이어질 수 있습니다. 우리는 이러한 편집 비효율을 해결하기 위해, AI를 활용한 자동 편집 기술의 필요성을 느꼈습니다.

# Introduction
 클라이밍의 짜릿한 순간들은 SNS를 통해 공유될 때 더욱 빛을 발합니다. 생동감 넘치는 등반 영상을 남기고 싶지만, 촬영을 부탁하거나 복잡한 편집 프로그램으로 씨름하는 과정은 때로 즐거움을 방해하기도 합니다.

 이제 그 수고로움을 저희에게 맡기세요. **CruxCut은 클라이머의 열정을 담은 영상을 손쉽게 완성시켜 드립니다.** 혁신적인 객체 검출 기술로 영상 속 당신을 정확히 찾아내고, 오직 당신에게 포커스된 하이라이트 영상을 자동으로 편집합니다. 더 이상 번거로운 편집에 시간을 낭비하지 마세요. 당신의 모든 등반이 곧바로 SNS를 빛낼 멋진 작품이 됩니다.

 (version2)본 프로젝트는 딥러닝 기반 객체 검출 기술을 활용하여, 영상 내 등반자를 자동으로 인식하고 중심 프레임을 선별함으로써, 실시간으로 등반 중심 장면만을 자동 추출·편집하는 AI 편집 서비스입니다.
YOLO 기반의 경량화된 객체 검출 모델을 활용하여 등반자의 위치를 실시간으로 추적하며, 해당 정보는 불필요한 장면을 제거하고 하이라이트 중심으로 영상을 재구성하는 데 사용됩니다. 이로써 편집자는 보다 효율적이고 직관적으로 클라이밍 콘텐츠를 제작할 수 있으며, 수작업 편집 시간을 획기적으로 줄일 수 있습니다. 이 서비스는 스포츠 영상 자동화 편집의 첫걸음으로, 향후 다양한 액티비티 영상에도 적용 가능성을 지니고 있습니다.

# Introductoin
 클라이밍 영상은 고정된 카메라로 원거리에서 촬영됩니다. 등반자가 움직임이 많아 영상내에서 작아지거나 다른 인물이 영상에 나타나 보는이로 하여금 산만함을 느끼게 합니다.(나에게 집중된 느낌이 들지 않습니다) 또한, 영상의 몰입감을 위해 수동 편집 도구를 사용하는 것은 시간과 노력이 많이 소모되는 작업입니다. (이러한 문제는 클라이밍 뿐만 아니라 스포츠, 공연 등 다양한 영상 제작자에게 생산성 저하와 콘텐츠 품질 저하를 유발합니다. 본 프로젝트는 이러한 편집 비효율을 해결하고자, AI를 활용한 자동 편집 기술인 crux-cut을 개발했습니다.

 crux-cut은 1분내의 영상을 YOLO를 통한 등반자 추적과 등반자 중심의 영상 재구성 과정을 자동 편집하는 AI 서비스입니다. crux-cut을 통해 사용자는 클라이밍 영상 편집 시간을 53% 감소시켜, 획기적인 편집 경험을 가질수 있습니다. 이러한 crux-cut의 자동 편집 서비스는 축구와 같은 원거리 스포츠 영상 자동 편집의 첫걸음으로, 향후 다양한 활동 영상에도 적용 가능성을 가지고 있습니다. 
 
## 주요 기능

-   **AI 기반 사용자 검출**: 고급 객체 검출 모델을 활용하여 영상 내 클라이머를 정확하게 인식하고 추적합니다.
-   **포커싱된 영상 생성**: 사용자가 항상 화면 중앙에 오도록 영상을 크롭하거나 이동시켜 몰입도를 높입니다.
-   **간편한 업로드/다운로드**: 원본 영상 업로드 및 편집된 영상 다운로드 기능을 직관적인 UI로 제공합니다.
-   **다양한 영상 포맷 지원**: MP4, AVI 등 주요 영상 포맷을 지원하여 사용성을 높입니다.

<p align="center">
:exclamation: CrxuCut AI가 클라이밍 영상에서 어떻게 여러분을 포착하는지 직접 확인하세요 :exclamation:
</p>

[![CruxCut tracks your move](https://img.youtube.com/vi/FbD5ZKpMjNA/maxresdefault.jpg)](https://www.youtube.com/shorts/FbD5ZKpMjNA) 

 # Demo
crux-cut은 직관적을 ui를 제공하며, 아래와 같은 과정을 통해 사용자에게 편집 효율을 경험시켜줍니다.
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
<div align="center">

권경범 사진 담당역할 이메일
정성훈 사진 담당역할 이메일
민건 사진 담당역할 이메일
</div>
