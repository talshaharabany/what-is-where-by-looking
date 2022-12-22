# What is Where by Looking: Weakly-Supervised Open-World Phrase-Grounding without Text Inputs
<p align="center">
  <img src="pics/pic1.PNG" width="800">
</p>

- Paper : [link](https://arxiv.org/abs/2206.09358)

- Demo : [link](https://replicate.com/talshaharabany/what-is-where-by-looking)

<p align="center">
  <img src="pics/pic2.PNG" width="800">
</p>

### Get Started
- datasets format are followed by [link](https://github.com/hassanhub/MultiGrounding/tree/master/data)

**For training a model :**
```
python train_grounding.py -bs 32 -nW 8 -nW_eval 1 -task vg_train -data_path /path_to/vg -val_path /path_to/flicker
python train_grounding.py -bs 32 -nW 8 -nW_eval 1 -task coco -data_path /path_to/coco -val_path /path_to/flicker
```

**For Grounding evaluation with our model [XX is the number of the results folder i.e 'gpu22' - XX == 22]:**
```
python inference_grounding.py -task grounding -dataset refit -val_path /path_to/RefIt -Isize 224 -clip_eval 0 -path_ae XX -nW 1
python inference_grounding.py -task grounding -dataset flicker -val_path /path_to/flicker -Isize 224 -clip_eval 0 -path_ae XX -nW 1
python inference_grounding.py -task grounding -dataset vg -val_path /path_to/VG -Isize 224 -clip_eval 0 -path_ae XX -nW 1
```

**For Grounding evaluation with CLIP model:**
```
python inference_grounding.py -task grounding -dataset refit -val_path /path_to/RefIt -Isize 224 -clip_eval 1 -nW 1
python inference_grounding.py -task grounding -dataset flicker -val_path /path_to/flicker -Isize 224 -clip_eval 1 -nW 1
python inference_grounding.py -task grounding -dataset vg -val_path /path_to/VG -Isize 224 -clip_eval 1 -nW 1
```

**For WWbL evaluation with our model:**
```
python inference_grounding.py -task app -dataset refit -val_path /path_to/RefIt -Isize 224 -clip_eval 0 -path_ae XX -nW 1 --start 0 --end 9983
python wwbl_algo1_point_metric.py -nW 1 -predictions_path YY -val_path /path_to/RefIt --dataset refit

python inference_grounding.py -task app -dataset flicker -val_path /path_to/flicker -Isize 224 -clip_eval 0 -path_ae XX -nW 1 -start 0 -end 1000
python wwbl_algo1_point_metric.py -nW 1 -predictions_path YY -val_path /path_to/flicker --dataset flicker

python inference_grounding.py -task app -dataset vg -val_path /path_to/VG -Isize 224 -clip_eval 0 -path_ae XX -nW 1 -start 0 -end 17478
python wwbl_algo1_point_metric.py -nW 1 -predictions_path YY -val_path /path_to/VG --dataset VG
```

<p align="center">
  <img src="pics/pic3.PNG" width="800">
</p>

### Phrase Grounding Results - Point Accuracy Metric

| Method | Backbone | VG(VGtrained/COCO) | Flicker(VGtrained/COCO) | ReferIt(VGtrained/COCO)  |  
| :---: |  :---:  |  :---:  |  :---:  |  :---:  |  
| Baseline | Random | 11.15 | 27.24 | 24.30 | 
| Baseline | Center | 20.55 | 47.40 | 30.30 | 
| GAE | CLIP | 54.72 | 72.47 | 56.76 | 
| FCVC | VGG |-/14.03 | -/29.03 | -/33.52 | 
| VGLS | VGG |24.40/- | -/- | -/- | 
| TD | Inception-2 | 19.31/- | 42.40/- | 31.97/- | 
| SSS | VGG | 30.03/- | 49.10/- | 39.98/- | 
| MG | BiLSTM+VGG | 50.18/46.99 | 57.91/53.29 | 62.76/47.89 | 
| MG | ELMo+VGG | 48.76/47.94 | 60.08/61.66 | 60.01/47.52 | 
| GbS | VGG | 53.40/52.00 | 70.48/72.60 | 59.44/56.10 | 
| **ours** | **CLIP+VGG** |**62.31/59.09** | **75.63/75.43** | **65.95/61.03** | 


<p align="center">
  <img src="pics/pic4.PNG" width="800">
</p>