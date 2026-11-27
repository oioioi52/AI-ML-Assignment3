# 인공지능과 머신러닝 과제 3: 3D Motion Data Post-Processing

영상 5개에서 MediaPipe 3D keypoint를 추출하고, 교수님이 제공한 Height Adjustment(Basic)와 직접 구현한 4단계 후처리(Enhanced)를 실험으로 비교한 과제입니다.

## 결과 요약 (5개 영상 평균, `results/methods_mean.csv`)

| 지표 | Raw | Basic(교수님) | Enhanced |
|---|---|---|---|
| Jerk (m/s³) | 625 | 687 | **122** |
| 뼈 길이 CV (%) | 9.26 | 15.11 | **0.00** |
| 골반 높이 떨림 (m/s²) | 0.01* | 6.29 | **0.64** |
| 남은 spike (개/영상) | 94.4 | 107.0 | **11.4** |
| 지면 관통 (%) | 38.59 | 0.00 | 0.11 |
| 동작 범위(ROM) 보존 (%) | 100 | 98.5 | 95.9 |

\* Raw는 골반이 원점에 고정돼 있어 높이 정보가 없습니다.
정답 데이터가 없으므로 위 지표는 **안정성(부드러움·일관성)** 을 재는 것이고, 위치 정확도 향상을 뜻하지 않습니다.

## 폴더 구조

```
utils/viser_test.py            교수님 제공 PoseViser (수정 없음)
scripts/
  p1_create_raw_keypoints.py   영상 -> MediaPipe 3D keypoint (교수님 코드 기반)
  p1_2_post_processing.py      교수님 코드 + [추가] 표시한 Enhanced 후처리  <- 과제의 핵심 수정 파일
  motion_lib.py                후처리 알고리즘, 평가 지표
  run_experiments.py           E0~E6 비교, 파라미터 sweep, ablation, 강건성
  bvh_compare.py               DeepMotion BVH 와의 일치도 비교
  make_figures.py              보고서 그래프
  render_comparison_video.py   원본 | Raw | Basic | Enhanced 비교 영상
  build_viewer.py              브라우저용 3-way 뷰어 생성
  viser_compare.py             Viser 3-way 뷰어 (교수님 PoseViser 확장)
  build_report.py              교수님 PPT 양식에 결과를 채워 PPTX 생성
data/
  raw_keypoints/*.npz          MediaPipe 원본 (교수님 npz 형식 + fps, 2D landmark)
  processed/*_raw|_target|_enhanced.npy   (T,33,3)   *_target = Basic.  같은 이름의 .csv 는 Unity용
  reference_bvh/*.bvh          DeepMotion 결과(비교 참고용, 정답 아님)
  robustness/                  1/2, 1/4 해상도로 다시 추출한 keypoint
results/                       모든 표(csv), 실험 로그, figures/
videos/                        blender_character_*.mp4 (리깅 캐릭터 비교). 원본 화면이 들어가는 compare_*.mp4 는 레포에 없음 -> render_comparison_video.py 로 생성
visualization/
  motion_viewer.html           더블클릭으로 여는 비교 뷰어
  blender_compare.py           리깅 캐릭터 3명 비교 (Blender)
  (캐릭터 FBX 는 레포에 없음: visualization/assets/character_tpose.fbx 위치에 T-pose 리깅 캐릭터를 두고 실행)
```

## 교수님 코드에서 바꾼 부분

교수님이 주신 두 파일은 원본을 그대로 두고, 바꾼 줄에 `[수정]` / `[추가]` 주석을 달았습니다.

- `p1_create_raw_keypoints.py`: YouTube 다운로드 대신 로컬 영상을 읽도록 `main()` 수정, 영상마다 Pose 객체 생성, fps·2D landmark 추가 저장. `download_youtube()`와 keypoint 저장 형식은 그대로입니다.
- `p1_2_post_processing.py`: `create_height_corrected_target()` 등 기존 함수는 그대로, `enhanced_pipeline()`과 저장 코드 몇 줄만 추가했습니다.
- 나머지 파일(`motion_lib.py` 등)은 새로 만든 파일입니다.

## 실행 방법

```bash
conda activate ai_ml_final          # 또는 pip install -r requirements.txt
python scripts/p1_create_raw_keypoints.py --model_complexity 1   # data/raw_videos/*.mp4 필요
python scripts/p1_2_post_processing.py
python scripts/run_experiments.py
python scripts/bvh_compare.py
python scripts/make_figures.py
python scripts/render_comparison_video.py
python scripts/build_viewer.py
python scripts/build_report.py --template <교수님 PPT 양식 경로> --name "<학번 이름>"
blender --background --python visualization/blender_compare.py -- --video handstand --render --mp4
```

난수를 쓰지 않으므로 같은 입력이면 항상 같은 결과가 나옵니다.

## 알아둘 점

- **MediaPipe 모델**: 저장된 결과는 `--model_complexity 1`(full)로 추출했습니다. 스크립트 기본값은 교수님 코드와 같은 2(heavy)이며, 2로 다시 추출하면 수치가 달라질 수 있습니다.
- **원본 영상**: 원본 영상과 원본 화면이 들어간 그림·영상은 올리지 않습니다(.gitignore). keypoint 파일(`data/raw_keypoints`)부터 실행하면 2번 단계 이후를 재현할 수 있습니다.
- **Unity**: CSV는 교수님 `export_unity_csv()`로 출력했습니다. 캐릭터 시각화는 Blender로 했습니다.
- **BVH**: DeepMotion 결과는 다른 추정기의 출력이므로 정답으로 쓰지 않고 일치도만 봤습니다.
