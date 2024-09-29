import jsonpickle
import json
from datetime import datetime
import os, random

canteens = [
  "桃李园一层", "桃李园二层",
  "紫荆园一层", "紫荆园二层", "紫荆园三层", "紫荆园四层", 
  "观畴园一层", "观畴园二层",
  "丁香园",
  "清青快餐",
  "清芬园一层", "清芬园二层",
  "玉树园",
  "芝兰园",
  "南园",
  "听涛园正厅",
  "听涛园侧厅",
  "荷园一层",
  "荷园二层",
  "澜园",
  "双清园",
]

b_canteens = [
  "桃李园", "紫荆园", "听涛园", "清芬园", "观畴园", "不吃了",
]

e_canteens = [
  "桃李园", "玉树园", "点外卖", "出学校吃", "不吃了",
]

random.shuffle(b_canteens)
breakfast = b_canteens[:3]
random.shuffle(canteens)
lunch = canteens[:3]
random.shuffle(canteens)
dinner = canteens[:3]
random.shuffle(e_canteens)
late_night_snack = e_canteens[:3]
    

breakfast_data = {
  "schemaVersion": 1,
  "label": "早餐",
  "message": f"{' '.join(breakfast)}",
}
lunch_data = {
  "schemaVersion": 1,
  "label": "午餐",
  "message": f"{' '.join(lunch)}",
}
dinner_data = {
  "schemaVersion": 1,
  "label": "晚餐",
  "message": f"{' '.join(dinner)}",
}
late_night_snack_data = {
  "schemaVersion": 1,
  "label": "夜宵",
  "message": f"{' '.join(late_night_snack)}",
}
os.makedirs('results', exist_ok=True)
with open(f'../_data/results/breakfast.json', 'w') as outfile:
    json.dump(breakfast_data, outfile, ensure_ascii=False)
with open(f'../_data/results/lunch.json', 'w') as outfile:
    json.dump(lunch_data, outfile, ensure_ascii=False)
with open(f'../_data/results/dinner.json', 'w') as outfile:
    json.dump(dinner_data, outfile, ensure_ascii=False)
with open(f'../_data/results/lns.json', 'w') as outfile:
    json.dump(late_night_snack_data, outfile, ensure_ascii=False)
