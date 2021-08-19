```shell
# 198
nohup python train.py -u > DIN-base-seq.log 2>&1 &  # 1.5 min/epoch
# need to edit test.py `test_path`
168546
nohup python test.py > test.log 2>&1 & # about 10min

#din_weights.epoch_0005.val_loss_0.0858.ckpt

pip install -r requirements.txt
pip freeze > requirements.txt
```


## Data

```
orders_poi_session.txt
wm_order_id	clicks	dt

orders_train.txt
user_id	wm_order_id	wm_poi_id	aor_id	order_price_interval	order_timestamp	ord_period_name	order_scene_name	aoi_id	takedlvr_aoi_type_name	dt

orders_spu_train.txt

oders_test_poi.txt
user_id	wm_order_id	aor_id	order_timestamp	ord_period_name	aoi_id	takedlvr_aoi_type_name	dt
```


## DIN

```
Input: 
# dense_inputs and sparse_inputs is empty
# seq_inputs (None, maxlen, behavior_num)
# item_inputs (None, behavior_num)

Output:
(n, 1)
```
