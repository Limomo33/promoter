from idt_fl_cl_fftda import *
# from keras.utils import np_utils
# from sklearn.manifold import TSNE
# from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA
from keras import backend as K
from keras.layers import add,Activation,Concatenate
from sklearn.cluster import AgglomerativeClustering,KMeans
from evlo import *
import re
from keras.utils import np_utils
from sklearn.manifold import TSNE
#tf.compat.v1.enable_eager_execution()
def batch_norm_ff(xs,xt):
    epsilon = 1e-3
    ave=np.load('E:/text/Dataset/mammal/Hu_Mou_aver.npy')
    va=np.load('E:/text/Dataset/mammal/Hu_Mou_var.npy')

    alpha=ave
    beta=va

    srmean, srvariance = K.mean(xs), K.var(xs)
    trmean, trvariance = K.mean(xt), K.var(xt)

    scale, shift = K.ones_like(srmean), K.zeros_like(srmean)
    x = (xt - (alpha*srmean+(1-alpha)*trmean)) / K.sqrt(beta*srvariance + (1-beta)*trvariance+epsilon)
    return scale * x + shift

def batch_norm_ft(xt,xs):
    epsilon = 1e-3
    fp=np.load('E:/text/Dataset/mammal/Hu_Mou_fp.npy')
    theta=np.load('E:/text/Dataset/mammal/Hu_Mou_theta.npy')

    alpha=fp
    beta=theta

    srmean, srvariance = K.mean(xs), K.var(xs)
    trmean, trvariance = K.mean(xt), K.var(xt)

    scale, shift = K.ones_like(srmean), K.zeros_like(srmean)
    x = (xt - (alpha*srmean+(1-alpha)*trmean)) / K.sqrt(beta*srvariance + (1-beta)*trvariance+epsilon)
    return scale * x + shift
# def trans_dis(c,d,I):
#     shape=d.shape.as_list()
#     l=I.shape.as_list()[0]
#
#     for i in range(shape[0]):
#         cc=re.split("\t", c[i])
#         for j in cc:
#             if j>l:
#

# def Agg(data):
#     clus_ff = AgglomerativeClustering(n_clusters=2, distance_threshold=0, affinity='mantattan', linkage='single',
#                                       compute_full_tree=True).fit_predict(data)
#     return clus_ff



def get_dis(y1,y2):
    #alpha=0.3

    return tf.multiply(y1,y2)
def get_kl(p,q):
    return K.sum(p * K.log(p / q), axis=-1)
def t3_model(tr_ff,tr_ft,sr_ff,sr_ft):
    feature_size = 32
    nb_classes = 2

    inputs_1 = Input(shape=tr_ff)
    inputs_2 = Input(shape=tr_ft)
    inputs_3 = Input(shape=sr_ff)
    inputs_4 = Input(shape=sr_ft)

    xf = Lambda(lambda x: batch_norm_ff(*x))([inputs_3, inputs_1])
    xt = Lambda(lambda x: batch_norm_ft(*x))([inputs_4, inputs_2])


    c1 = Concatenate()([inputs_1, xf])

    c1 = Dense(feature_size, activation='relu')(c1)

    # att=Attention(2,8)([inputs_1,inputs_2,inputs_1])
    # print(K.int_shape(att))
    # att=AveragePooling1D()(att)
    c1 = Dropout(0.5)(c1)


    c2 = Concatenate()([inputs_2, xt])


    c2 = Dense(feature_size, activation='relu')(c2)
    c2 = Dropout(0.5)(c2)
    # 防止过拟合
    c1 = Flatten()(c1)
    c2 = Flatten()(c2)
    c = Concatenate()([c1, c2])
    feature = Dense(feature_size, activation='relu')(c)
    # predict_o = Dense(nb_classes, activation='softmax',name='softmax')(feature)
    predict = Dense(nb_classes, activation=None, name='arc')(feature)
    predict_o = Softmax(name='softmax')(predict)
    #
    input_target = Input(shape=(1,))
    centers = Embedding(nb_classes, feature_size)(input_target)  # Embedding层用来存放中心损失
    l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')([feature, centers])
    print(K.int_shape(l2_loss))

    # arcLoss = Lambda(lambda x: arcface_loss(*x), name='arcLoss')([input_target, predict])
    # arcface = ArcFace(name='arc_face')([predict, input_target])

    optr = Adam(0.0001)#b"0.002
    # model_train = Model(inputs=[inputs_1,inputs_2, input_target],outputs=[predict, l2_loss,arcLoss])
    # model_train = Model(inputs=[inputs_1, inputs_2, input_target], outputs=[predict_o,arcface,l2_loss])
    model_train = Model(inputs=[inputs_1, inputs_2, inputs_3,inputs_4,input_target], outputs=[predict_o, l2_loss])
    model_train.compile(optimizer=optr, loss=[focal_loss_fixed, lambda y_true, y_pred: y_pred],
                        loss_weights=[1., 0.5],
                        metrics={'softmax': 'accuracy'})  # 将一个字符串编译为字节代码
    # loss_weights=[1., 0.5,0.1]
    model_train.summary()
    return model_train

def t2_model(tr_ff,tr_ft,sr_ff,sr_ft):
    feature_size = 32
    nb_classes = 2

    inputs_1 = Input(shape=tr_ff)
    inputs_2 = Input(shape=tr_ft)
    inputs_3 = Input(shape=sr_ff)
    inputs_4 = Input(shape=sr_ft)

    xf = Lambda(lambda x: batch_norm_ff(*x))([inputs_3, inputs_1])
    xt = Lambda(lambda x: batch_norm_ft(*x))([inputs_4, inputs_2])

    conv12 = Conv1D(32, kernel_size=5, strides=1, padding="same")(xf)  # 第三层卷积模块，卷积核为3，特征图个数为64；粗粒度特征提取b:5
    conv12 = LeakyReLU(alpha=0.2)(conv12)
    conv12 = Dropout(0.5)(conv12)
    # f12 = Flatten()(conv12)  # 输出展平
    c1 = Concatenate()([inputs_1, conv12])

    c1 = Dense(feature_size, activation='relu')(c1)
    # att=Attention(2,8)([inputs_1,inputs_2,inputs_1])
    # print(K.int_shape(att))
    # att=AveragePooling1D()(att)
    # conv = Dropout(0.25)(att)

    conv22 = Conv1D(32, kernel_size=5, strides=1, padding="same")(xt)  # 第三层卷积模块，卷积核为3，特征图个数为64；粗粒度特征提取
    conv22 = LeakyReLU(alpha=0.2)(conv22)
    conv22 = Dropout(0.5)(conv22)
    # f22 = Flatten()(conv22)  # 输出展平
    c22 = Concatenate()([inputs_2, conv22])


    c2 = Dense(feature_size, activation='relu')(c22)
    # 防止过拟合
    c1 = Flatten()(c1)
    c2 = Flatten()(c2)
    c = Concatenate()([c1, c2])
    feature = Dense(feature_size, activation='relu')(c)
    # predict_o = Dense(nb_classes, activation='softmax',name='softmax')(feature)
    predict = Dense(nb_classes, activation=None, name='arc')(feature)
    predict_o = Softmax(name='softmax')(predict)
    #
    input_target = Input(shape=(1,))
    centers = Embedding(nb_classes, feature_size)(input_target)  # Embedding层用来存放中心损失
    l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')([feature, centers])
    print(K.int_shape(l2_loss))

    # arcLoss = Lambda(lambda x: arcface_loss(*x), name='arcLoss')([input_target, predict])
    # arcface = ArcFace(name='arc_face')([predict, input_target])

    optr = Adam(0.0001)#b"0.002
    # model_train = Model(inputs=[inputs_1,inputs_2, input_target],outputs=[predict, l2_loss,arcLoss])
    # model_train = Model(inputs=[inputs_1, inputs_2, input_target], outputs=[predict_o,arcface,l2_loss])
    model_train = Model(inputs=[inputs_1, inputs_2, inputs_3,inputs_4,input_target], outputs=[predict_o, l2_loss])
    model_train.compile(optimizer=optr, loss=[focal_loss_fixed, lambda y_true, y_pred: y_pred],
                        loss_weights=[1., 0.3],
                        metrics={'softmax': 'accuracy'})  # 将一个字符串编译为字节代码
    # loss_weights=[1., 0.5,0.1]
    model_train.summary()
    return model_train

def t_model(tr_ff,tr_ft,sr_ff,sr_ft):
    feature_size = 32
    nb_classes = 2

    inputs_1 = Input(shape=tr_ff)
    inputs_2 = Input(shape=tr_ft)
    inputs_3 = Input(shape=sr_ff)
    inputs_4 = Input(shape=sr_ft)

    xf = Lambda(lambda x: batch_norm_ff(*x))([inputs_3, inputs_1])
    xt = Lambda(lambda x: batch_norm_ft(*x))([inputs_4, inputs_2])

    conv12 = Conv1D(32, kernel_size=5, strides=1, padding="same")(xf)  # 第三层卷积模块，卷积核为3，特征图个数为64；粗粒度特征提取b:5
    conv12 = LeakyReLU(alpha=0.2)(conv12)
    conv12 = Dropout(0.5)(conv12)
    # f12 = Flatten()(conv12)  # 输出展平
    c12 = Concatenate()([inputs_1, conv12])
    conv12 = Conv1D(32, kernel_size=5, strides=1, padding="same")(c12)  # 第三层卷积模块，卷积核为3，特征图个数为64；粗粒度特征提取b:5
    conv12 = LeakyReLU(alpha=0.2)(conv12)
    conv12 = Dropout(0.5)(conv12)
    c13 = Concatenate()([c12, conv12])
    # c1=Concatenate()([c11,c12])
    c1 = Dense(feature_size, activation='relu')(c13)
    # att=Attention(2,8)([inputs_1,inputs_2,inputs_1])
    # print(K.int_shape(att))
    # att=AveragePooling1D()(att)
    # conv = Dropout(0.25)(att)

    conv22 = Conv1D(32, kernel_size=5, strides=1, padding="same")(xt)  # 第三层卷积模块，卷积核为3，特征图个数为64；粗粒度特征提取
    conv22 = LeakyReLU(alpha=0.2)(conv22)
    conv22 = Dropout(0.5)(conv22)
    # f22 = Flatten()(conv22)  # 输出展平
    c22 = Concatenate()([inputs_2, conv22])
    conv22 = Conv1D(32, kernel_size=5, strides=1, padding="same")(c22)  # 第三层卷积模块，卷积核为3，特征图个数为64；粗粒度特征提取
    conv22 = LeakyReLU(alpha=0.2)(conv22)
    conv22 = Dropout(0.5)(conv22)
    # f22 = Flatten()(conv22)  # 输出展平
    c23 = Concatenate()([c22, conv22])

    c2 = Dense(feature_size, activation='relu')(c23)
    # 防止过拟合
    c1 = Flatten()(c1)
    c2 = Flatten()(c2)
    c = Concatenate()([c1, c2])
    feature = Dense(feature_size, activation='relu')(c)
    # predict_o = Dense(nb_classes, activation='softmax',name='softmax')(feature)
    predict = Dense(nb_classes, activation=None, name='arc')(feature)
    predict_o = Softmax(name='softmax')(predict)
    #
    input_target = Input(shape=(1,))
    centers = Embedding(nb_classes, feature_size)(input_target)  # Embedding层用来存放中心损失
    l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')([feature, centers])
    print(K.int_shape(l2_loss))

    # arcLoss = Lambda(lambda x: arcface_loss(*x), name='arcLoss')([input_target, predict])
    # arcface = ArcFace(name='arc_face')([predict, input_target])

    optr = Adam(0.0001)#b"0.002
    # model_train = Model(inputs=[inputs_1,inputs_2, input_target],outputs=[predict, l2_loss,arcLoss])
    # model_train = Model(inputs=[inputs_1, inputs_2, input_target], outputs=[predict_o,arcface,l2_loss])
    model_train = Model(inputs=[inputs_1, inputs_2, inputs_3,inputs_4,input_target], outputs=[predict_o, l2_loss])
    model_train.compile(optimizer=optr, loss=[focal_loss_fixed, lambda y_true, y_pred: y_pred],
                        loss_weights=[1., 0.3],
                        metrics={'softmax': 'accuracy'})  # 将一个字符串编译为字节代码
    # loss_weights=[1., 0.5,0.1]
    model_train.summary()
    return model_train

def test_model():

    file1 = 'E:/text/Dataset/mammal/human/'
    sp1='mammal'
    sr=['human-250+50_positive']
    file2 = 'E:/text/Dataset/mammal/mouse/'
    sp2='mammal'
    tr=['mouse-250+50_positive','mouse+100+400_negative']
    train_X_sr, train_y_sr, test_X_sr, test_y_sr = load_onehot_data_sele(file1, sr,sp1)
    train_X_tr, train_y_tr, test_X_tr, test_y_tr = load_onehot_data_sele(file2, tr,sp2)
    train_X_DPCP_sr = train_X_sr[:, :, :9]
    test_X_DPCP_sr = test_X_sr[:, :, :9]
    train_X_FFT_sr = train_X_sr[:, :, 9:]
    test_X_FFT_sr = test_X_sr[:, :, 9:]
    train_X_DPCP_tr = train_X_tr[:, :, :9]
    test_X_DPCP_tr = test_X_tr[:, :, :9]
    train_X_FFT_tr = train_X_tr[:, :, 9:]
    test_X_FFT_tr = test_X_tr[:, :, 9:]
    print('----------------------------------------------')
    print(train_X_DPCP_tr.shape, train_X_FFT_sr.shape)
    train_y_sr = train_y_sr.reshape((-1, 1))
    test_y_sr = test_y_sr.reshape((-1, 1))  # reshape 转换为一列
    train_y_tr = train_y_tr.reshape((-1, 1))
    test_y_tr = test_y_tr.reshape((-1, 1))  # reshape 转换为一列
    print('-------------------------------------------')
    print(test_X_sr.shape, test_y_sr.shape)
    train_y_oh_sr = OneHotEncoder().fit_transform(train_y_sr.reshape((-1, 1))).todense()
    test_y_oh_sr = OneHotEncoder().fit_transform(test_y_sr.reshape((-1, 1))).todense()
    train_y_oh_tr = OneHotEncoder().fit_transform(train_y_tr.reshape((-1, 1))).todense()
    test_y_oh_tr = OneHotEncoder().fit_transform(test_y_tr.reshape((-1, 1))).todense()
    shape_fft = train_X_FFT_sr.shape[1:]
    shape_DPCP = train_X_DPCP_sr.shape[1:]
    # print(len(shape_fft))


    base_model = DCGAN_D(shape_fft, shape_DPCP)
    #base_model.summary()
    base_model.load_weights('E:/text/cl_fl_cl_ffthuman.h2')
    for layer in base_model.layers:
        layer.trainable=False
    #[X,_,_]=base_model.output
    #get_layer1=K.function([base_model_1.input],[base_model_1.layers('flatten_1').output,base_model_1.layers('flatten_2').output])
    ##############################l1_6,7(None, 300, 64);l2,14,15 (None, 300, 32);l3,22,23(None, 300, 32)######################
    get_srFF_layer = Model(input=base_model.input, outputs=base_model.layers[22].output)
    get_srFt_layer=Model(input=base_model.input, outputs=base_model.layers[23].output)
    sr_ff=get_srFF_layer.predict([train_X_FFT_sr,train_X_DPCP_sr, np.ones(len(train_X_sr))])
    sr_ft=get_srFt_layer.predict([train_X_FFT_sr,train_X_DPCP_sr, np.ones(len(train_X_sr))])

    get_trFF_layer= Model(input=base_model.input, outputs=[base_model.layers[22].output])
    get_trFt_layer = Model(input=base_model.input, outputs=base_model.layers[23].output)
    # random_y = np.ones((len(train_X)))
    # sr_ff=get_srFF_layer.output
    # sr_ft=get_srFt_layer.output#([inputs_1,inputs_2,input_target_sr])
    # tr_ff=get_trFF_layer.output#([inputs_3,inputs_4,input_target_tr])
    # tr_ft= get_trFt_layer.output#([inputs_3, inputs_4, input_target_tr])
    #sr_ff=get_FF_layer.predict([train_X_FFT_sr,train_X_DPCP_sr, np.ones(len(train_X_sr))])
    #sr_ft=get_Ft_layer.predict([train_X_FFT_sr,train_X_DPCP_sr, np.ones(len(train_X_sr))])
    tr_ff=get_trFF_layer.predict([train_X_FFT_tr,train_X_DPCP_tr, np.ones(len(train_X_tr))])
    sr_ff=np.vstack((sr_ff,sr_ff))[:len(tr_ff)]
    sr_ft=np.vstack((sr_ft,sr_ft))[:len(tr_ff)]
    tr_ft=get_trFt_layer.predict([train_X_FFT_tr,train_X_DPCP_tr, np.ones(len(train_X_tr))])

    print('Finish pre-train data reading')
    print(tr_ff.shape)#(40676, 300, 72)
    model_train = t2_model(tr_ff.shape[1:],tr_ft.shape[1:],sr_ff.shape[1:],sr_ft.shape[1:])
    random_y = np.ones((len(train_X_tr)))

    checkpoint = ModelCheckpoint('E:/text/model/transfermodel/human_mou2.h2', monitor='val_softmax_acc', verbose=1,
                                 save_best_only=True, mode='max', period=1)
    model_train.fit([tr_ff, tr_ft, sr_ff,sr_ft,train_y_tr], [train_y_oh_tr, random_y],  # [train_y_oh_tr,train_y_oh_tr,random_y]
                    batch_size=128,  validation_split=0.3, epochs=200,
                    callbacks=[checkpoint])

    model_train.load_weights('E:/text/model/transfermodel/human_mou2.h2')
    # [pred] = model_train.predict([test_X_FFT, test_X_DPCP])  # ,np.ones(len(test_X))])
    sr_ff_test = get_srFF_layer.predict([test_X_FFT_sr, test_X_DPCP_sr, np.ones(len(test_X_sr))])
    sr_ft_test = get_srFt_layer.predict([test_X_FFT_sr, test_X_DPCP_sr, np.ones(len(test_X_sr))])
    tr_ff_test = get_trFF_layer.predict([test_X_FFT_tr, test_X_DPCP_tr, np.ones(len(test_X_tr))])
    tr_ft_test = get_trFt_layer.predict([test_X_FFT_tr, test_X_DPCP_tr, np.ones(len(test_X_tr))])
    sr_ff_test = np.vstack((sr_ff_test, sr_ff_test))[:len(tr_ff_test)]
    sr_ft_test = np.vstack((sr_ft_test, sr_ft_test))[:len(tr_ft_test)]

    [pred_y_fl, loss2] = model_train.predict([tr_ff_test, tr_ft_test,sr_ff_test,sr_ft_test, np.ones(len(tr_ff_test))])
    # dense1_layer_model = tf.keras.models.Model(inputs=model_train.input, outputs=model_train.get_layer('arcface_loss_output').output)
    # pred_y = dense1_layer_model.predict(x=[test_X_FFT,test_X_DPCP, np.ones(len(test_X))])

    # pred_y=model_train.load_weights('cl_fl_cl_fft.h2')
    # pred_y = np.zeros(len(test_X))
    l = [pred_y_fl]
    for ii in range(len(l)):
        pred_y = l[ii]
        class_y = np.zeros((len(test_X_tr), 6))
        print('第' + str(ii) + '个loss')
        for i in range(len(test_X_tr)):
            # pred_y[i]=np.argmax(pred[i])
            class_y[i][np.argmax(pred_y[i])] = 1
            ##########idt#############
        # test_acc = metrics.accuracy_score(test_y, pred_y)
        # test_m = metrics.matthews_corrcoef(test_y, pred_y)
        # precision = metrics.precision_score(test_y, pred_y)
        # recall = metrics.recall_score(test_y, pred_y)
        # tnr_score = tnr(test_y, pred_y)
        # print('test_acc:{},{},{},{},{}'.format(test_acc, test_m, precision, recall, tnr_score))
        #########################################
        target_names = ['0', '1']
        SP = np.zeros((2, 1))
        SN = np.zeros((2, 1))
        MCC = np.zeros((2, 1))
        Acc = np.zeros((2, 1))
        result = []
        for i, t in enumerate(target_names):
            t_test_y = test_y_oh_tr[:, i]
            t_pred_y = class_y[:, i]
            target_name = ['0', t]
            test_acc = metrics.accuracy_score(t_test_y, t_pred_y)
            Acc[i] = test_acc
            test_m = metrics.matthews_corrcoef(t_test_y, t_pred_y)
            MCC[i] = test_m
            result.append(test_m)
            precision = metrics.precision_score(t_test_y, t_pred_y)
            recall = metrics.recall_score(t_test_y, t_pred_y)
            SN[i] = recall
            tnr_score = tnr(t_test_y, t_pred_y)
            SP[i] = tnr_score

            print('type:{},test_acc:{},{},{},{},{}'.format(t, test_acc, test_m, precision, recall, tnr_score))
        # print(metrics.classification_report(test_y, class_y, target_names=target_names))
    # print(sr_ff)
    # print(sr_ff.shape)
    # KLd=Lambda(lambda x: get_kl(x[0],x[1]))
    # kl_f=KLd([inputs_1,inputs_3])
    # kl_t=KLd([inputs_2,inputs_4])
    # kl=add([kl_f,kl_t])
    #print(K.eval(kl))
    # ff=Concatenate()([sr_ff,tr_ff])
    # #ff=Dense()
    # ft=Concatenate()([sr_ft,tr_ft])
    #Input_d = Input(shape=D.shape[1])
    #dis=Lambda(get_dis(Input_d,kl))
    #FF = tf.constant(ff, dtype=tf.float32)
    # Convert the tensor to a NumPy array
    #ff = ff.numpy()
    # ff=np.concatenate([sr_ff,tr_ff])
    # ft=np.concatenate([sr_ft,tr_ft])
    # print(ff.shape)
    # clus_ff=AgglomerativeClustering(n_clusters=2,distance_threshold=None,affinity='precomputed', linkage='single',compute_full_tree=True)
    # clu=clus_ff.fit_predict(D)
    # print(clu)
    # clus_ft = AgglomerativeClustering(n_clusters=2,distance_threshold=None, affinity='euclidean', linkage='single',
    #                                   compute_full_tree=True).fit_predict(ft)

    #clus=clu
    #input_1=Input(shape=(1,))

    # y_clusff=clus_ff.distance_
    # y_clusft = clus_ft.distance_
    # # print(K.eval(clus_ft))
    # y_chil=clus_ft.children_
    # # print(K.eval(y_chil))
    # print("rfffffffffflabel",np.reduce_max(clus_ff.label_))
    # print("rttttttttttlabel", np.reduce_max(clus_ft.label_))


    # y_clusff=Activation('Relu')(y_clusff)
    # y_clusft=Activation('Relu')(y_clusft)
    #xf=Lambda(batch_norm(inputs_3,inputs_1,inputs_5))
    #xt = Lambda(batch_norm(inputs_4, inputs_2, inputs_6))
    #
    # x=Concatenate()([xf,xt])
    # x=Flatten()(x)
    # predict = Dense(2, activation='softmax')(x)
    #
    # model_tran = Model(input=[get_srFF_layer.input,get_srFF_layer.input,get_trFF_layer.input,get_trFt_layer.input], outputs=predict)
    # model_tran.compile(optimizer=optr, loss='categorical_crossentropy', metrics=['accuracy'])
    # checkpoint = ModelCheckpoint('/data1/lrm1/text/1.h2', monitor='val_acc', verbose=1,
    #                              save_best_only=True, mode='max', period=1)
    # model_tran.fit([train_X_FFT_tr, train_X_DPCP_tr,train_y_oh_sr],  # [train_y_oh,random_y]
    #                 batch_size=128, validation_split=0.3, epochs=200,
    #                 callbacks=[checkpoint])

    # pred_y = model_tran.predict([test_X_FFT,test_X_DPCP, np.ones(len(test_X))])
    # class_y = np.zeros((len(test_X), 2))
    # for i in range(len(test_X)):
    #     # pred_y[i]=np.argmax(pred[i])
    #     class_y[i][np.argmax(pred_y[i])] = 1
    #     ##########idt#############
    # # test_acc = metrics.accuracy_score(test_y, pred_y)
    # # test_m = metrics.matthews_corrcoef(test_y, pred_y)
    # # precision = metrics.precision_score(test_y, pred_y)
    # # recall = metrics.recall_score(test_y, pred_y)
    # # tnr_score = tnr(test_y, pred_y)
    # # print('test_acc:{},{},{},{},{}'.format(test_acc, test_m, precision, recall, tnr_score))
    # #########################################
    # target_names = ['0', '1']
    # SP = np.zeros((2, 1))
    # SN = np.zeros((2, 1))
    # MCC = np.zeros((2, 1))
    # Acc = np.zeros((2, 1))
    # result = []
    # for i, t in enumerate(target_names):
    #     t_test_y = test_y_oh[:, i]
    #     t_pred_y = class_y[:, i]
    #     target_name = ['0', t]
    #     test_acc = metrics.accuracy_score(t_test_y, t_pred_y)
    #     Acc[i] = test_acc
    #     test_m = metrics.matthews_corrcoef(t_test_y, t_pred_y)
    #     MCC[i] = test_m
    #     result.append(test_m)
    #     precision = metrics.precision_score(t_test_y, t_pred_y)
    #     recall = metrics.recall_score(t_test_y, t_pred_y)
    #     SN[i] = recall
    #     tnr_score = tnr(t_test_y, t_pred_y)
    #     SP[i] = tnr_score
    #
    #     print('type:{},test_acc:{},{},{},{},{}'.format(t, test_acc, test_m, precision, recall, tnr_score))

def load_weight():
    train_X, train_y, test_X, test_y = load_onehot_data()
    train_X_DPCP = train_X[:, :, :9]
    test_X_DPCP = test_X[:, :, :9]
    train_X_FFT = train_X[:, :, 9:]
    test_X_FFT = test_X[:, :, 9:]
    print('----------------------------------------------')
    print(train_X_DPCP.shape, train_X_FFT.shape)
    train_y = train_y.reshape((-1, 1))
    test_y = test_y.reshape((-1, 1))  # reshape 转换为一列
    print('-------------------------------------------')
    print(test_X.shape, test_y.shape)
    train_y_oh = OneHotEncoder().fit_transform(train_y.reshape((-1, 1))).todense()
    test_y_oh = OneHotEncoder().fit_transform(test_y.reshape((-1, 1))).todense()
    shape_fft = train_X_FFT.shape[1:]
    shape_DPCP = train_X_DPCP.shape[1:]
    # print(len(shape_fft))
#####################################################################################################
    random_y = np.ones((len(train_X)))
    y = train_y.astype(int)
    base_model = DCGAN_D(shape_fft, shape_DPCP)
    base_model.load_weights('E:/text/cl_fl_cl_ffthuman.h2')
    [X, _, _] = base_model.output
    predict = Dense(2, activation='softmax')(X)
    model_tran = Model(input=base_model.input, outputs=predict)
    optr = Adam(0.001)
    model_tran.compile(optimizer=optr, loss='categorical_crossentropy', metrics=['accuracy'])
    model_tran.load_weights('E:/text/model/bug_worm.h2')
    train_feature=model_tran.predict([train_X_FFT, train_X_DPCP, train_y])
    np.save('E:/text/model/bug_worm.npy',train_feature)
#####################################################################################################
def tsne_view(train_feature,y):
    # -------------------------------PCA,tSNE降维分析--------------------------------
    # pca = PCA(n_components=2)  # 总的类别
    # pca_result = pca.fit_transform(train_feature)
    # print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))

    # Run T-SNE on the PCA features.
    tsne = TSNE(n_components=3, verbose=1)  # 降维后的维度
    tsne_results = tsne.fit_transform(train_feature)

    # -------------------------------可视化--------------------------------

    y_test_cat = np_utils.to_categorical(y, num_classes=4)  # 总的类别
    color_map = np.argmax(y_test_cat, axis=1)
    plt.figure(figsize=(15, 15))
    for cl in range(4):  # 总的类别
        indices = np.where(color_map == cl)
        indices = indices[0]
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=cl,alpha=0.5)
    plt.legend()
    plt.savefig('MHC_fl.png')
    plt.show()


def view():
    label=[]
    data=[]
    #sigmas=['bug_fruit_fly','bug_honey_bee','bug_worm','fish_zebrafish']
    sigma= ['human', 'mouse1', 'Rat', 'rhesus_macaque']
    #sigmas=['dog-250+50_positive','mouse-250+50_positive','Rat-250+50_positive','rhesus_macaque-250+50_positive']#,'chicken-250+50_positive']

    for s in range(len(sigma)):
        D=np.load('E:/text/Dataset/mammal/'+sigma[s]+'_fl.npy')
        label += [[s] for _ in range(len(D))]
        if data != []:
            data = np.concatenate((data, D))
        else:
            data = D
    label=np.array(label)
    data=np.array(data)
    tsne_view(data,label)

if __name__=="__main__":
    #load_weight()
    test_model()