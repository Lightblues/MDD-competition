## Data

```
orders_poi_session.txt
wm_order_id	clicks	dt

orders_train.txt
user_id	wm_order_id	wm_poi_id	aor_id	order_price_interval	order_timestamp	ord_period_name	order_scene_name	aoi_id	takedlvr_aoi_type_name	dt

orders_spu_train.txt
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
