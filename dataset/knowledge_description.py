import openai
import json
import os

openai.api_key = "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"

def gen(categoty,comps):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": 'You will help me to describe object in natural language according to the given name of object and it\'s components. \
        And each component description should include Color, Shape, Quantity, and one Special Attribute. And you should give me \
        the position relationship and adjacency for all components. Please return as a JSON file following format shown below and keep consistency all the time: \
        { \
            "Bird": { \
            "Head":      {"Color":"yellow", \
                         "Shape":"round", \
                         "Quantity":"1", \
                         "Special Attribute":"There is a sharp beak front of the head"} \
            "Body":      {"Color":"white,gray", \
                         "Shape":"streamline", \
                         "Quantity":"1", \
                         "Special Attribute":"None"} \
            "Wing":      {"Color":"gray,black", \
                         "Shape":"wide and round", \
                         "Quantity":"2", \
                         "Special Attribute":"None"} \
            ...} \
            Relationship: \
            [ \
                "Head is/up/to Body", \
                "Head is/next/to Body", \
                "Wing is/next/to Body", \
                "Wing is/side/of Body", \
                "Tail is/behind/of Body", \
                ... \
            ] }\
        '},
        {"role": "user", "content": 'The target is {} with components: {}'.format(categoty,str(comps))}
            ],
        temperature=0.0,
        max_tokens=3200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['message']['content']    

sup_cat_2_seg = {
    "Quadruped":["Head", "Body", "Foot", "Tail"],               # 四足动物
    "Biped":["Head", "Body", "Hand", "Foot", "Tail"],           # 二足动物
    "Fish" :["Head", "Body", "Fin", "Tail"],                    # 鱼
    "Bird" :["Head", "Body", "Wing", "Foot", "Tail"],           # 鸟
    "Snake" :["Head", "Body"],                                  # 蛇
    "Reptile" :["Head", "Body", "Foot", "Tail"],                # 爬虫
    "Car" :["Body", "Tier", "Side Mirror"],                     # 车
    "Bicycle" :["Head", "Body", "Seat", "Tier"],                # 自行车
    "Boat" :["Body", "Sail"],                                   # 船
    "Aeroplane" :["Head", "Body", "Wing", "Engine", "Tail"],    # 飞机
    "Bottle" :["Body", "Mouth"]                                 # 瓶子
}

spcat_ch_2_en = {
    "猴子" : "Biped",
    "狗" : "Quadruped",
    "羊" : "Quadruped",
    "鱼" : "Fish",
    "蛇" : "Snake",
    "飞机" : "Aeroplane",
    "药丸" : "Bottle",
    "牛" : "Quadruped",
    "吉普车" : "Car",
    "鸟" : "Bird",
    "鳄鱼" : "Reptile",
    "青蛙" : "Reptile",
    "猩猩" : "Biped",
    "猫" : "Quadruped",
    "瓶子" : "Bottle",
    "蜥蜴" : "Reptile",
    "房车" : "Car",
    "帆船" : "Boat",
    "熊" : "Quadruped",
    "壁虎" : "Reptile",
    "赛车" : "Car",
    "龟" : "Quadruped",
    "骆驼" : "Quadruped",
    "狒狒" : "Biped",
    "蛤蟆" : "Reptile",
    "老虎" : "Quadruped",
    "警车" : "Car",
    "狼" : "Quadruped",
    "拖拉机" : "Car",
    "双人自行车" : "Bicycle",
    "鹅" : "Bird",
    "啤酒" : "Bottle",
    "鹰" : "Bird",
    "铲雪车" : "Car",
    "狐狸" : "Quadruped",
    "狸" : "Quadruped",
    "校车" : "Car",
    "水瓶" : "Bottle",
    "高尔夫车" : "Car",
    "豹" : "Quadruped",
    "轿车" : "Car",
    "出租车" : "Car",
    "松鼠" : "Quadruped",
    "小客车" : "Car",
    "儿童自行车" : "Bicycle",
    "摩托车" : "Bicycle",
    "熊猫" : "Quadruped",
    "卡丁车" : "Car",
    "自行车" : "Bicycle",
    "猪" : "Quadruped",
    "运动型轿车" : "Car",
    "救护车" : "Car",
    "独轮车" : "Bicycle",
    "葡萄酒" : "Bottle",
    "加长车" : "Car",
    "鲸鱼" : "Fish",
    "卡车" : "Car",
    "助动车" : "Car",
    "巴士" : "Car",
    "船" : "Boat",
    "面包车" : "Car",
}

f = open('supcat_and_fine_cat.txt','r')

obj_list = []
for line in f.readlines():
    obj_entry = {}
    info = line.split(',')
    comps = sup_cat_2_seg[spcat_ch_2_en[info[0]]]
    object = info[-1]
    obj_entry['obj'] = object
    obj_entry['comps'] = comps
    obj_list.append(obj_entry)

desc = []
idx = 0
for e in obj_list:
    idx += 1
    if (os.path.exists(os.path.join("temp_gen",str(idx)+".json"))):
        f = open(os.path.join("temp_gen",str(idx)+".json"),'r')
        desc.append(json.loads(f.read()))
        continue
    res = gen(e['obj'],e['comps'])
    res = json.loads(res)
    print (res)
    with open(os.path.join("temp_gen",str(idx)+".json"),'w') as f:
        f.write(json.dumps(res))
    f.close()
    desc.append(res)

# f = open("PartImageNet_Desc.txt",'w')
# for c in desc:
#     f.write(c+'\n')
# f.close()

f = open("PartImageNet_Desc.json",'w')
f.write(json.dumps(desc))
f.close()