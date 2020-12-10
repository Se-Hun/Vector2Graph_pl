# Vector2Graph_pl
심층신경망 언어이해에서의 벡터-그래프 변환 방법을 통한 설명가능성 확보에 대한 연구(허세훈, 정상근)(2020)(제 32회 한글 및 한국어 정보처리 학술대회 논문집)

## 필요한 라이브러리 설치
* conda install numpy
* conda install pandas
* conda install matplotlib
* conda install networkx

* pytorch --> [pytorch 설치](https://pytorch.org/get-started/locally/)
* conda install pytorch-lightning
* pip install transformers

## 다음과 같은 순서로 실행한다.
### 데이터 준비
* `data/DOMAIN_NAME/` 경로에 json 형태로 된 nlu 데이터를 준비한다.
* `data/` 경로에서 `python 99_build_all.py --domain weather`와 같은 형식으로 실험을 위한 전처리된 데이터를 생성한다.

### [TEXT] BERT 기반의 의도분류 모델 훈련
* `python run_intent_classification.py --domain weather --do_train --text_reader bert`와 같이 의도분류 실험을 위한 기본 Text Reader를 훈련시킨다.

### Vector Movement Generator
* `python run_intent_classification.py --domain weather --do_test_and_dump --text_reader bert --num_samples 50`와 같이 Dropout을 통한 Perturbated Vector Movement를 생성한다.
* 이 때, num_samples는 같은 Input에 대해 Duplicate(복제)되는 수이다. 이 Sample들은 같은 입력 값을 가짐에도 Dropout에 의해 미묘하게 다른 Output Vector가 생성된다.

### Vector To Graph Converter
* `python vector2graph_converter.py --domain weather --text_reader bert --num_samples 50 --rep_type graph --need_edges --top_n 10`와 같이 생성된 Vector Movement를 Graph Image 형태로 변환한다.
* 이 때, need_edges 옵션을 주면 graph의 edge가 표현되고 주지 않으면 edge는 표현되지 않는다.
* 또한, top_n은 그래프를 생성하기 위한 node의 수를 의미한다.

### [Image] CNN 기반의 의도분류 실험
* `python image_intent_classification.py --domain weather --do_train --do_test --image_reader cnn --num_samples 50 --need_edges --base_text_reader bert --rep_type graph --top_n 10`을 통해 CNN 기반의 의도분류 실험을 할 수 있다.

## 실험 성능
![fig1 : 실험성능](figs/experiment_result.PNG)

## 그래프 변환 예
![fig2 : 그래프 변환 예](figs/compare_graph_image.PNG)