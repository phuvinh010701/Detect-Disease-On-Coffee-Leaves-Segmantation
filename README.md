# Semantic Segmantation

---

## Data labeling

<div align="center" style="display: inline_block"><br>

<img src='storage/tool_icon.jpg' style='width: 5%'>

Tool: [MVTec Deeplearning Tool 22.06](https://www.mvtec.com/products/deep-learning-tool) 

<img src='storage/tool_label.png' style='width: 50%'>
</div>


### Label: Sau ve bua
<div align="center" style="inline_block">
<img src='storage/label_sauvebua.png' style='width: 50%'>
</div>

### Label: Nam ri sat
<div align="center" style="inline_block">
<img src='storage/label_namrisat.jpg' style='width: 50%'>
</div>

### Overview:

- 534 images
- Sau ve bua: 327 objects
- Nam ri sat: 531 objects

<div align="center" style="inline_block">
<img src='storage/overview.png' style='width: 50%'>
</div>

## Build model unet

## Without augmented
### Version 1

<div align="center" style="inline_block">
<img src='storage/model_ver1.png' style='width: 50%'>
</div>

* num parameters: 7.699.011

* Accuracy on test set: 0.9669

<div align="center" style="inline_block">
<img src='storage/res_ver1_1.png' style='width: 50%'>
<img src='storage/res_ver1_2.png' style='width: 50%'>
<img src='storage/res_ver1_3.png' style='width: 50%'>
<img src='storage/res_ver1_4.png' style='width: 50%'>
<img src='storage/res_ver1_5.png' style='width: 50%'>
</div>

* Kết quả dự đoán ở class 1 chưa tốt
