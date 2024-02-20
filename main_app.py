import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

def get_model():
  input_layer = tf.keras.Input(shape=(256, 256, 3)) # インプット層

  x = tf.keras.layers.Rescaling(1 / 255)(input_layer) # スケーリング層
  x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x) # 畳み込み層 : 特徴を抽出
  x = tf.keras.layers.MaxPooling2D(2)(x) # 余計な情報を排除する　（小さい値は排除される）
  x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x) # 畳み込み層 : 特徴を抽出
  x = tf.keras.layers.MaxPooling2D(2)(x) # 余計な情報を排除する　（小さい値は排除される）
  x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x) # 畳み込み層 : 特徴を抽出
  x = tf.keras.layers.MaxPooling2D(2)(x) # 余計な情報を排除する　（小さい値は排除される）
  x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x) # 畳み込み層 : 特徴を抽出
  x = tf.keras.layers.MaxPooling2D(2)(x) # 余計な情報を排除する　（小さい値は排除される）
  x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x) # 畳み込み層 : 特徴を抽出
  x = tf.keras.layers.MaxPooling2D(2)(x) # 余計な情報を排除する　（小さい値は排除される）
  x = tf.keras.layers.Flatten()(x) # 一列にする
  x = tf.keras.layers.Dense(64, activation='relu')(x) # 適当に62個のニューロン用意
  output_layer = tf.keras.layers.Dense(3, activation='softmax')(x) # 出力層：三つゲートあり、一番大きい値が該当するラベルを示す

  model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

  return model

def predict(model, image, n_classes):
  prediction = model(np.array([image]))  # [0.4, 0.5, 0.9]

  predicted_class = n_classes[np.argmax(prediction)] # 例えば、 healthy
  confidence = 100 * np.max(prediction)

  return predicted_class, confidence


st.title('ポテト疫病診断')

upload_file = st.file_uploader("画像をアップロードしてください")

if upload_file is not None: # 何かアップロードされると
    
    st.image(upload_file)  # アップロードされた画像は何か？
    
    # データ形式を変換
    pil_image = Image.open(upload_file)
    numpy_image = np.array(pil_image)

    # 確かめる
    #st.write(upload_file)

    #st.write(numpy_image)
    
    # モデルを準備
    model = get_model()
    model.load_weights('weights/model_weight') # ノーブックで学習したモデルを読み取った

    prediction, confidence = predict(model, numpy_image, ['疫病早期', '疫病末期', '健康'])

    st.write(prediction)
    st.write(confidence)