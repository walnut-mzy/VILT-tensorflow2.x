import tensorflow as tf
from tqdm import *

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer,BertModel,BertConfig
import numpy as np
import setting
import cv2
class Dataset:
    def __init__(self):
        self.path_data=setting.path_data
        self.path_data_img=setting.path_img
        self.label=[]
        self.place_list=[]
        self.txt_list=[]
        self.img_list=[]
        self.img_data=[]
        self.place_data=[]
        self.bert_tokenize=[]
        self.dataset_len=None
        with open(self.path_data,"r",encoding="GBK") as fp:
            data=fp.read().split("\n")
            self.dataset_len=len(data)-1
            for i in data:
                if i=="":
                    continue
                dic=eval(i)
                self.label.append(dic["scenelabel"])
                self.place_list.append([float(dic["longitude"]),float(dic["latitude"])])
                self.txt_list.append(dic["describe"])
                self.img_list.append(dic["image"])
        assert self.dataset_len != None, "The dataset is empty, or there is a problem somewhere in the dataset"
        self.class_num = len(set(self.label))
        self.label=tf.one_hot(self.label,depth=len(set(self.label)))
        self.len_val_tokenize=None
        self.len_test_tokenize = None
        self.len_train_tokenize=None
        self.tokenizer = BertTokenizer.from_pretrained(setting.tokenize_path)
        self.vocab_size=len(self.tokenizer.vocab)
        # self.txt_data=self.tokenizer(self.txt_list, truncation=True, padding=True, max_length=setting.max_size)
        # self.input_ids=self.txt_data["input_ids"]
        # self.token_type_ids=self.txt_data["token_type_ids"]
        # self.attention_mask=self.txt_data["attention_mask"]
        # print(len(self.input_ids),len(self.token_type_ids),len(self.attention_mask))
    def __len__(self):
        return self.dataset_len
    def normailze(self,list_track):
        """
        :param list_track: 需要归一化的坐标
        :return: 列表
        """
        gps_x = np.array([i[0] for i in list_track], dtype=float)
        gps_y = np.array([i[1] for i in list_track], dtype=float)

        gps_x_max = float(max(gps_x))
        gps_y_max = float(max(gps_y))
        gps_x_min = float(min(gps_x))
        gps_y_min = float(min(gps_y))

        gps_x = (gps_x - gps_x_min) / (gps_x_max - gps_x_min)
        gps_y = (gps_y - gps_y_min) / (gps_y_max - gps_y_min)

        return [[i, j] for i, j in zip(gps_x.tolist(), gps_y.tolist())]
    def img_get(self,img_path):
        """
        :param img_path: 这个img_path只是个名字如：19在img_get这个函数中完成拼接等任务
        :return: 处理过的图片样式
        """
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.IMREAD_COLOR)
        return cv2.resize(image, (setting.img_w, setting.img_h))
    def data_process(self):
        """

        :return: 返回处理后的数据集
        """
        self.place_data=self.normailze(self.place_list)
        flag=1
        for i in range(self.dataset_len):
            self.img_data.append(self.img_get(self.path_data_img+"/"+str(self.img_list[i])+".jpg")/255.0+1e-3)

        # data_list=[{
        #     "place": self.place_data[i],
        #     "img": self.img_data[i],
        #     "txt": {
        #         "input_ids":self.input_ids[i],
        #         "token_type_ids":self.token_type_ids[i],
        #         "attention_mask":self.attention_mask[i]
        #     }
        # } for i in range(self.dataset_len)]


        x_train_place,x_test_place,x_train_img_data,x_test_img_data,x_train_txt_list,x_test_txt_list,y_train,y_test= train_test_split(
            self.place_data,self.img_data,self.txt_list, self.label.numpy(), test_size=0.2)

        x_test_txt_list=self.tokenizer(x_test_txt_list,truncation=True, padding=True, max_length=setting.max_size)
        if setting.is_val:
            x_train_place, x_val_place, x_train_img_data, x_val_img_data, x_train_txt_list, x_val_txt_list, y_train, y_val = train_test_split(
               x_train_place, x_train_img_data, x_train_txt_list, y_train, test_size=0.2)

            x_val_txt_list = self.tokenizer(x_val_txt_list, truncation=True, padding=True,
                                              max_length=setting.max_size)
            self.len_val_tokenize=tf.convert_to_tensor(x_val_txt_list["input_ids"]).shape[1]
            dataset_val = tf.data.Dataset.from_tensor_slices(({
                                                                  "place":x_val_place,
                                                                  "img":x_val_img_data,
                                                                  "input_ids":x_val_txt_list["input_ids"],
                                                                  "token_type_ids": x_val_txt_list["token_type_ids"],
                                                                  "attention_mask":x_val_txt_list["attention_mask"]
                                                              }, y_val)).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeeat)

            flag=0
        x_train_txt_list = self.tokenizer(x_train_txt_list, truncation=True, padding=True,
                                          max_length=setting.max_size)

        self.len_train_tokenize=tf.convert_to_tensor(x_train_txt_list["input_ids"]).shape[1]
        dataset_train = tf.data.Dataset.from_tensor_slices(({
                                                              "place": x_train_place,
                                                              "img": x_train_img_data,
                                                            "input_ids": x_train_txt_list["input_ids"],
                                                            "token_type_ids": x_train_txt_list["token_type_ids"],
                                                            "attention_mask": x_train_txt_list["attention_mask"]
                                                          }, y_train)).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)
        self.len_test_tokenize=tf.convert_to_tensor(x_test_txt_list["input_ids"]).shape[1]
        dataset_test = tf.data.Dataset.from_tensor_slices(({
                                                                "place": x_test_place,
                                                                "img": x_test_img_data,
                                                               "input_ids": x_test_txt_list["input_ids"],
                                                               "token_type_ids": x_test_txt_list["token_type_ids"],
                                                               "attention_mask": x_test_txt_list["attention_mask"]
                                                            }, y_test)).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)


        if setting.is_val:
            print("验证数据集的长度为:", len(dataset_val)*setting.batch)
            print(dataset_val)
            print("训练数据集长度为:", len(dataset_train)*setting.batch)
            print(dataset_train)
            print("测试数据集长度为:", len(dataset_test)*setting.batch)
            print(dataset_test)
            return dataset_train,dataset_test,dataset_val
        print("训练数据集长度为:", len(dataset_train)*setting.batch)
        print(dataset_train)
        print("测试数据集长度为:", len(dataset_test)*setting.batch)
        print(dataset_test)
        return dataset_train,dataset_test
    def data_process_unbert(self):
        """

        :return: 返回处理后的数据集
        """
        self.place_data=self.normailze(self.place_list)

        for i in range(self.dataset_len):
            self.img_data.append(self.img_get(self.path_data_img+"/"+str(self.img_list[i])+".jpg")/255.0+1e-3)

        # data_list=[{
        #     "place": self.place_data[i],
        #     "img": self.img_data[i],
        #     "txt": {
        #         "input_ids":self.input_ids[i],
        #         "token_type_ids":self.token_type_ids[i],
        #         "attention_mask":self.attention_mask[i]
        #     }
        # } for i in range(self.dataset_len)]


        x_train_place,x_test_place,x_train_img_data,x_test_img_data,x_train_txt_list,x_test_txt_list,y_train,y_test= train_test_split(
            self.place_data,self.img_data,self.txt_list, self.label.numpy(), test_size=0.2)

        x_test_txt_list=self.tokenizer(x_test_txt_list,truncation=True, padding=True, max_length=setting.max_size)
        if setting.is_val:
            x_train_place, x_val_place, x_train_img_data, x_val_img_data, x_train_txt_list, x_val_txt_list, y_train, y_val = train_test_split(
               x_train_place, x_train_img_data, x_train_txt_list, y_train, test_size=0.2)

            x_val_txt_list = self.tokenizer(x_val_txt_list, truncation=True, padding=True,
                                              max_length=setting.max_size)
            self.len_val_tokenize=tf.convert_to_tensor(x_val_txt_list["input_ids"]).shape[1]
            dataset_val = tf.data.Dataset.from_tensor_slices(({
                                                                  "place":x_val_place,
                                                                  "img":x_val_img_data,
                                                                  "txt":x_val_txt_list["input_ids"],
                                                              }, y_val)).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)


        x_train_txt_list = self.tokenizer(x_train_txt_list, truncation=True, padding=True,
                                          max_length=setting.max_size)

        self.len_train_tokenize=tf.convert_to_tensor(x_train_txt_list["input_ids"]).shape[1]
        dataset_train = tf.data.Dataset.from_tensor_slices(({
                                                              "place": x_train_place,
                                                              "img": x_train_img_data,
                                                            "txt": x_train_txt_list["input_ids"]
                                                          }, y_train)).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)
        self.len_test_tokenize=tf.convert_to_tensor(x_test_txt_list["input_ids"]).shape[1]
        dataset_test = tf.data.Dataset.from_tensor_slices(({
                                                                "place": x_test_place,
                                                                "img": x_test_img_data,
                                                               "txt": x_test_txt_list["input_ids"],
                                                            }, y_test)).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)


        if setting.is_val:
            print("验证数据集的长度为:", len(dataset_val)*setting.batch)
            print(dataset_val)
            print("训练数据集长度为:", len(dataset_train)*setting.batch)
            print(dataset_train)
            print("测试数据集长度为:", len(dataset_test)*setting.batch)
            print(dataset_test)
            return dataset_train,dataset_test,dataset_val
        print("训练数据集长度为:", len(dataset_train)*setting.batch)
        print(dataset_train)
        print("测试数据集长度为:", len(dataset_test)*setting.batch)
        print(dataset_test)
        return dataset_train,dataset_test
    def data_process_img_only(self):
        for i in range(self.dataset_len):
            self.img_data.append(self.img_get(self.path_data_img + "/" + str(self.img_list[i]) + ".jpg") / 255.0 + 1e-3)
        x_train_img_data, x_test_img_data,y_train, y_test = train_test_split(
           self.img_data, self.label.numpy(), test_size=0.2)
        if setting.is_val:
            x_train_img_data, x_val_img_data,y_train, y_val = train_test_split(
                x_train_img_data, y_train, test_size=0.2)
            dataset_val = tf.data.Dataset.from_tensor_slices((x_val_img_data, y_val)).shuffle(setting.BUFFER_SIZE).prefetch(
                tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)
        dataset_train = tf.data.Dataset.from_tensor_slices((x_train_img_data, y_train)).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)
        dataset_test = tf.data.Dataset.from_tensor_slices((x_test_img_data, y_test)).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)
        if setting.is_val:
            print("验证数据集的长度为:", len(dataset_val) * setting.batch)
            print(dataset_val)
            print("训练数据集长度为:", len(dataset_train) * setting.batch)
            print(dataset_train)
            print("测试数据集长度为:", len(dataset_test) * setting.batch)
            print(dataset_test)
            return dataset_train, dataset_test, dataset_val
        print("训练数据集长度为:", len(dataset_train) * setting.batch)
        print(dataset_train)
        print("测试数据集长度为:", len(dataset_test) * setting.batch)
        print(dataset_test)
        return dataset_train, dataset_test
    def data_process_txt_only(self):

        x_train_txt_data, x_test_txt_data, y_train, y_test = train_test_split(
            self.txt_list, self.label.numpy(), test_size=0.2)
        x_test_txt_list = self.tokenizer(x_test_txt_data, truncation=True, padding=True, max_length=setting.max_size)
        if setting.is_val:
            x_train_txt_data, x_val_txt_data, y_train, y_val = train_test_split(
                x_train_txt_data, y_train, test_size=0.2)
            x_val_txt_list = self.tokenizer(x_val_txt_data, truncation=True, padding=True,
                                            max_length=setting.max_size)
            self.len_val_tokenize = tf.convert_to_tensor(x_val_txt_list["input_ids"]).shape[1]
            dataset_val = tf.data.Dataset.from_tensor_slices(({
                                                                  "input_ids": x_val_txt_list["input_ids"],
                                                                  "token_type_ids": x_val_txt_list["token_type_ids"],
                                                                  "attention_mask": x_val_txt_list["attention_mask"]
                                                              }, y_val)).shuffle(setting.BUFFER_SIZE).prefetch(
                tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)
        x_train_txt_list = self.tokenizer(x_train_txt_data, truncation=True, padding=True,
                                          max_length=setting.max_size)
        self.len_train_tokenize=tf.convert_to_tensor(x_train_txt_list["input_ids"]).shape[1]
        dataset_train = tf.data.Dataset.from_tensor_slices(({
                                                                "input_ids": x_train_txt_list["input_ids"],
                                                                "token_type_ids": x_train_txt_list["token_type_ids"],
                                                                "attention_mask": x_train_txt_list["attention_mask"]
                                                            }, y_train)).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)
        self.len_test_tokenize = tf.convert_to_tensor(x_test_txt_list["input_ids"]).shape[1]
        dataset_test = tf.data.Dataset.from_tensor_slices(({
                                                               "input_ids": x_test_txt_list["input_ids"],
                                                               "token_type_ids": x_test_txt_list["token_type_ids"],
                                                               "attention_mask": x_test_txt_list["attention_mask"]
                                                           }, y_test)).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)
        if setting.is_val:
            print("验证数据集的长度为:", len(dataset_val) * setting.batch)
            print(dataset_val)
            print("训练数据集长度为:", len(dataset_train) * setting.batch)
            print(dataset_train)
            print("测试数据集长度为:", len(dataset_test) * setting.batch)
            print(dataset_test)
            return dataset_train, dataset_test, dataset_val
        print("训练数据集长度为:", len(dataset_train) * setting.batch)
        print(dataset_train)
        print("测试数据集长度为:", len(dataset_test) * setting.batch)
        print(dataset_test)
        return dataset_train, dataset_test