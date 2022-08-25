import os
import setting
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.activations import gelu
from transformers.models.bert.modeling_tf_bert import TFBertEmbeddings,BertConfig
# patch_embedding层，包括图片的embedding+分类头, 加上pos_embedding
from model.transformer import Transformer as placeclf
import os
import setting
import re
import sys
import GPUtil

if setting.is_gpu==False:
    print("CPU已被使用")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from train_only import main
from tensorflow.keras.layers import Layer,Conv2D,Input
from tensorflow.keras import Sequential, Model
if setting.txtclf!="bert":
    from model.concat_resnet152 import MultimodalConcatBertClf
else:
    from model.concat_bert import MultimodalConcatBertClf
from data import Dataset
if setting.is_gpu:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    z = GPUtil.getGPUs()
    print("使用的gpu为:",z[0].name)
    print("gpu负载率:", z[0].load)
    print("gpu显存:", z[0].memoryTotal)
    print("gpu驱动:", z[0].driver)
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
class PatchEmbedding(Layer):
    def __init__(self, image_size, patch_size, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)

        self.embed_dim = embed_dim
        self.n_patches = (image_size // patch_size) * (image_size // patch_size)
        self.patch_embed = Conv2D(self.embed_dim, patch_size, patch_size)

        # 添加分类的token,会concat到image_tokens中,使得shape为[b,196+1,768]
        self.cls_token = self.add_weight('cls_token', shape=[1, 1, self.embed_dim],
                                         dtype='float32', initializer='glorot_uniform',
                                         trainable=True)
        # pos_embedding与(image_tokens+cls_token)相加,所以shape也必须为[b,197,768]
        self.pos_embeding = self.add_weight('pos_embedding', shape=[1, self.n_patches + 1, self.embed_dim],
                                            dtype='float32', initializer='glorot_uniform',
                                            trainable=True)

    def call(self, inputs):
        # patch_size=16, embed_dim=768
        # [b,224,224,3] -> [b,14,14,768]
        x = self.patch_embed(inputs)
        # [b,14,14,768] -> [b,196,768]
        b, h, w, _ = x.shape
        x = tf.reshape(x, shape=[b, h * w, self.embed_dim])
        # 1,1,768 -> b,1,768
        cls_tokens = tf.broadcast_to(self.cls_token, (b, 1, self.embed_dim))
        # -> b, 197, 768
        x = tf.concat([x, cls_tokens], axis=1)

        # 加上pos_embedding -> b, 197, 728
        x = x + self.pos_embeding

        return x

    def get_config(self):
        config = super(PatchEmbedding, self).get_config()
        config.update({"embed_dim": self.embed_dim,
                       "num_patches": self.n_patches,
                       })
        return config


# msa层的实现
class multiHead_self_attention(Layer):
    def __init__(self, embed_dim, num_heads, attention_dropout=0.0, **kwargs):
        super(multiHead_self_attention, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.head_dim = embed_dim // self.num_heads
        self.all_head_dim = self.num_heads * self.head_dim

        self.scale = self.head_dim ** (-0.5)  # q*k之后的变换系数

        self.qkv = Dense(self.all_head_dim * 3)
        self.proj = Dense(self.all_head_dim)

        self.attention_dropout = Dropout(attention_dropout)

        self.softmax = Softmax()

    def call(self, inputs):
        # -> b,197,768*3
        qkv = self.qkv(inputs)
        # q,k,v: b,197,768
        q, k, v = tf.split(qkv, 3, axis=-1)

        b, n_patches, all_head_dim = q.shape
        # q,k,v: b,197,768 -> b,197,num_heads, head_dim 假设num_heads=12
        # b,197,768 -> b,197,12,64
        q = tf.reshape(q, shape=[b, n_patches, self.num_heads, self.head_dim])
        k = tf.reshape(k, shape=[b, n_patches, self.num_heads, self.head_dim])
        v = tf.reshape(v, shape=[b, n_patches, self.num_heads, self.head_dim])

        # b,197,12,64 -> b,12,197,64
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        # -> b,12,12,64
        attention = tf.matmul(q, k, transpose_b=True)
        attention = self.scale * attention
        attention = self.softmax(attention)
        attention = self.attention_dropout(attention)
        # -> b,12,197,64
        out = tf.matmul(attention, v)
        # b,12,197,64 -> b,197,12,64
        out = tf.transpose(out, [0, 2, 1, 3])
        # b,197,12,64 -> b,197,768
        out = tf.reshape(out, shape=[b, n_patches, all_head_dim])

        out = self.proj(out)
        return out

    def get_config(self):
        config = super(multiHead_self_attention, self).get_config()
        config.update({"num_heads": self.num_heads,
                       "head_dim": self.head_dim,
                       "all_head_dim": self.all_head_dim,
                       "scale": self.scale
                       })
        return config


class MLP(Layer):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.0, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

    def call(self, inputs):
        # 1,197,768 -> 1,197,768*4
        x = Dense(int(self.embed_dim * self.mlp_ratio))(inputs)
        x = gelu(x)
        x = Dropout(self.dropout)(x)

        # 1,197,768*4 - 1,197,768
        x = Dense(self.embed_dim)(x)
        x = Dropout(self.dropout)(x)

        return x

    def get_config(self):
        config = super(MLP, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout
        })
class ViLTransformer(Layer):
    def __init__(self,num_heads,patch_size,embed_dim,layer_length,num_classes,isencoder=True):
        super(ViLTransformer, self).__init__()
        self.layer_length=layer_length
        self.patchembedding=PatchEmbedding(setting.img_h,patch_size=patch_size,embed_dim=embed_dim,name="patchAndPos_embedding")


       # setting.config pretrain/bert/config.json
        bert_config = BertConfig.from_json_file(setting.config)
        self.txt_embedding = TFBertEmbeddings(bert_config)
        self.norms_1=[LayerNormalization(name=f"layernorm{i}_1") for i in range(layer_length)]
        self.attentions=[multiHead_self_attention(embed_dim,num_heads,0,name=f"MSA{i}") for i in range(layer_length)]
        self.norms_2 = [LayerNormalization(name=f"layernorm{i}_2") for i in range(layer_length)]
        self.MLPS=[MLP(embed_dim=embed_dim,name=f"MLP{i}") for i in range(layer_length)]
        self.Dn1=Dense(num_classes,name="classifier",activation="softmax")
        self.dn2=Dense(768,activation="relu")
        self.isencoder=isencoder
    def call(self, inputs):
        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]
        img_content = inputs["img"]

        x_img=self.patchembedding(img_content)
        x_img=self.dn2(x_img)
        x_txt=self.txt_embedding(input_ids,token_type_ids,attention_mask)

        x=tf.concat([x_img,x_txt],axis=1)
        for i in range(self.layer_length):
            x=self.norms_1[i](x)
            x1=self.attentions[i](x)

            x2=tf.concat([x,x1],axis=-1)
            x=self.norms_2[i](x)
            x=self.MLPS[i](x)

            x=tf.concat([x,x2],axis=-1)
        cls_token=x[:,0]
        if self.isencoder != True:
            # 1,768 -> 1, num_classes
            out = self.Dn1(cls_token)

        else:
            out = cls_token
        return out
class myCallback(tf.keras.callbacks.Callback):
    def __init__(self,path):
        self.fp = path

    def on_epoch_end(self, epoch, logs):
        print(epoch)
        #logs如下
        """
        {'loss': 0.35939720273017883, 'accuracy': 0.5134039521217346, 'val_loss': 0.2836693525314331, 'val_accuracy': 0.476856529712677}
        """
        """
        写入loss等基本信息
        epch,loss,accuracy,val_loss,val_accuracy
        """
        fp=open(self.fp, "a+")
        fp.write(str(epoch)+","+str(logs["loss"])+","+str(logs["accuracy"])+"\n")
        fp.close()
if __name__ == '__main__':
    dataset = Dataset()

    if setting.is_val:
        if setting.txtclf != 'bert':
            dataset_train, dataset_test, dataset_val = dataset.data_process_unbert()
        else:
            dataset_train, dataset_test, dataset_val = dataset.data_process()
    else:
        if setting.txtclf != 'bert':
            dataset_train, dataset_test = dataset.data_process_unbert()
        else:
            dataset_train, dataset_test = dataset.data_process()
    len_val_tokenize = dataset.len_val_tokenize
    len_test_tokenize = dataset.len_test_tokenize
    len_train_tokenize = dataset.len_train_tokenize
    class_num = int(dataset.class_num)
    print(class_num)
    vocab_size = dataset.vocab_size
    inputs = Input(shape=(224, 224, 3), batch_size=setting.batch)
    inputs2 = tf.keras.layers.Input(shape=(len_train_tokenize), dtype=tf.int32, batch_size=setting.batch)
    inputs3 = tf.keras.layers.Input(shape=(len_train_tokenize), dtype=tf.int32, batch_size=setting.batch)
    inputs4 = tf.keras.layers.Input(shape=(len_train_tokenize), dtype=tf.int32, batch_size=setting.batch)
    inputs5 = tf.keras.layers.Input(shape=(2), dtype=tf.float32, batch_size=setting.batch)
    input={
        "place":inputs5,
        "img":inputs,
        "input_ids":inputs2,
       "token_type_ids":inputs3,
       "attention_mask":inputs4,

    }
    vitmodel =ViLTransformer(num_heads=setting.num_heads,patch_size=setting.patch_size,embed_dim=setting.embed_dim,layer_length=setting.layer_length,num_classes=class_num,isencoder=False)
    out=vitmodel(input)
    print(out)
    model = Model(inputs=input, outputs=out, name='tf2-vit')
    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-08, clipnorm=1)
    train_loss = tf.keras.losses.CategoricalCrossentropy()
    train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')
    callbacks = [
        # 模型保存
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(setting.save_path, setting.imageclf + setting.txtclf + "{epoch}.h5"),
            monitor="accuracy",
            save_weights_only=True,
            verbose=1,
            period=10,
            save_best_only=True,
            mode="max",

        ),
        # tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        #                                  patience=20,
        #                                  restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(
            log_dir=setting.log_dir, histogram_freq=0, write_graph=True, write_images=False,
            update_freq='epoch', profile_batch=2, embeddings_freq=0,
            embeddings_metadata=None,
        ),
        myCallback("log1.txt")
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00000001)
    ]
    model.compile(
        optimizer=optimizer,
        loss=train_loss,
        metrics=['accuracy']
    )

    model.fit(
        dataset_train,
        epochs=setting.epoch,
        batch_size=setting.batch,  # initial_epoch=setting.initial_epoch,
        # validation_data=dataset_val,
        callbacks=callbacks,
    )


