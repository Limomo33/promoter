from idt_fl_cl_fft import *
def onehot(seqs,label):
    dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    sigmas=['sigma24promoter', 'sigma28promoter', 'sigma32promoter', 'sigma38promoter', 'sigma54promoter',
              'sigma70promoter']
    # print(seqs)
    matrix_onehot = np.zeros((len(seqs), len(seqs[0]), 4))
    matrix_MPB = np.zeros((len(seqs), len(seqs[0]), 4))
    # print(len(seqs[0]))
    # print(len(seqs))
    for i, s in enumerate(seqs):
        MPB = np.load(sigmas[label[i]] + '.npy')
        for j, bp in enumerate(s):
            for val in ['A', 'C', 'G', 'T']:
                if bp == val:
                    matrix_onehot[i, j, dict[bp.upper()]] = 1
                    matrix_MPB[i, j, dict[bp.upper()]] = MPB[dict[bp.upper()]][j]
                    break
                # else:
                #     matrix[i, j, 4] = 0
    return [np.array(matrix_onehot), np.transpose(matrix_MPB, (0, 2, 1))]
def NCP_DPCP_fea(seqs):
    res=[]
    NCP = np.zeros((len(seqs), len(seqs[0]), 3))
    DPCP=np.zeros((len(seqs),len(seqs[0]),6))
    dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    NCP_code=[[1,1,1],[0,0,1],[1,0,0],[0,1,0]]
    DPCP_code = {'AA': [0.5773884923447732, 0.6531915653378907, 0.6124592000985356, 0.8402684612384332, 0.5856582729115565,
                   0.5476708282666789],
            'AT': [0.7512077598863804, 0.6036675879079278, 0.6737051546096536, 0.39069870063063133, 1.0,
                   0.76847598772376],
            'AG': [0.7015450873735896, 0.6284296628760702, 0.5818362228429766, 0.6836002897416182, 0.5249586459219764,
                   0.45903777008667923],
            'AC': [0.8257018549087278, 0.6531915653378907, 0.7043281318652126, 0.5882368974116978, 0.7888705476333944,
                   0.7467063799220581],
            'TA': [0.3539063797840531, 0.15795248106354978, 0.48996729107629966, 0.1795369895818257, 0.3059118434042811,
                   0.32686549630327577],
            'TT': [0.5773884923447732, 0.6531915653378907, 0.0, 0.8402684612384332, 0.5856582729115565,
                   0.5476708282666789],
            'TG': [0.32907512978081865, 0.3312861433089369, 0.5205902683318586, 0.4179453841534657, 0.45898067049412195,
                   0.3501900760908136],
            'TC': [0.5525570698352168, 0.6531915653378907, 0.6124592000985356, 0.5882368974116978, 0.49856742124957026,
                   0.6891727614587756],
            'GA': [0.5525570698352168, 0.6531915653378907, 0.6124592000985356, 0.5882368974116978, 0.49856742124957026,
                   0.6891727614587756],
            'GT': [0.8257018549087278, 0.6531915653378907, 0.7043281318652126, 0.5882368974116978, 0.7888705476333944,
                   0.7467063799220581],
            'GG': [0.5773884923447732, 0.7522393476914946, 0.5818362228429766, 0.6631651908463315, 0.4246720956706261,
                   0.6083143907016332],
            'GC': [0.5525570698352168, 0.6036675879079278, 0.7961968911255676, 0.5064970193495165, 0.6780274730118172,
                   0.8400043540595654],
            'CA': [0.32907512978081865, 0.3312861433089369, 0.5205902683318586, 0.4179453841534657, 0.45898067049412195,
                   0.3501900760908136],
            'CT': [0.7015450873735896, 0.6284296628760702, 0.5818362228429766, 0.6836002897416182, 0.5249586459219764,
                   0.45903777008667923],
            'CG': [0.2794124572680277, 0.3560480457707574, 0.48996729107629966, 0.4247569687810134, 0.5170412957708868,
                   0.32686549630327577],
            'CC': [0.5773884923447732, 0.7522393476914946, 0.5818362228429766, 0.6631651908463315, 0.4246720956706261,
                   0.6083143907016332]}
    for i, s in enumerate(seqs):
        for j, bp in enumerate(s):
            NCP[i, j, :] = NCP_code[dict[bp.upper()]]

            if j ==0:
                DPCP[i,j,:]=DPCP_code[''.join([s[j], s[j + 1]])]
            elif j==len(s)-1:
                DPCP[i,j,:]=DPCP_code[''.join([s[j-1],s[j]])]
            else:
                key2 = ''.join([s[j], s[j + 1]])
                key1 = ''.join([s[j - 1], s[j]])
                DPCP[i, j, :] = np.divide((np.array(DPCP_code[key1])+np.array(DPCP_code[key2])),2)
    item = np.concatenate([DPCP, NCP],axis=2)
    # item=np.array([FFT_real,FFT_imag])

    #return np.transpose(np.array(item),(0,2,1))
    return np.array(item)
def load_onehot_data():
    #sigmas = ['negative2860_delate','positive2860_delate']
    #add = 'E:/text/Dataset/Independent_test_dataset/'
    add = 'E:/text/Dataset/mammal/chicken/'
    sigmas = 'independent'
    dict={'24':0,'28':1,'32':2,'38':3,'54':4,'70':5}
    seq1 = []
    label=[]
    seq=[]
    file=add+sigmas+'.txt'
    #file = 'E:/text/Dataset/Benchmark_dataset/promoter_and_non-promoter/'
    #file='/data1/lrm1/text/Dataset/Benchmark_dataset/promoter_and_non-promoter/'
    f=open(file, 'r',encoding='UTF-8').readlines()
    ele_basic = ['A', 'C', 'G', 'T']
    for l in f:
        if l[0]=='>':
            la=l.strip('>').strip()
            label+=[dict[la]]
        else:
            seq1.append(l.strip())
    for l in seq1:
        for e in range(len(l.strip())):
            if l[e] not in ele_basic:
                l=list(l)
                l[e] = random.sample(list(ele_basic), 1)
                l=''.join(str(l))
        if len(l)==80:
            l=l
        elif len(l)<80:
            ii=50-len(l)
            l+='X'*ii
        elif len(l)>80:
            l=l[:80]
        seq.append(l)

    [seq1, seq1m] = onehot(seq, label)
    spec1 = Spectrum_features(seq1m)
    pro1=NCP_DPCP_fea(seq)
    fea1=np.concatenate((spec1,pro1,seq1),axis=2)
    label = np.array(label)
    seq = np.array(fea1)
    return label, seq
def test_model():
    Y, X=load_onehot_data()
    X_DPCP = X[:, :, :9]
    X_FFT = X[:, :, 9:]
    y = Y.reshape((-1, 1))
    test_y = OneHotEncoder().fit_transform(y.reshape((-1, 1))).todense()
    l=np.zeros((len(X),1))
    test_y_oh=np.concatenate((test_y[:,:4],l),axis=1)
    test_y_oh=np.concatenate((test_y_oh,test_y[:,4:]),axis=1)
    shape_fft = X_FFT.shape[1:]
    shape_DPCP = X_DPCP.shape[1:]
    model_train = DCGAN_D(shape_fft, shape_DPCP)
    model_train.load_weights('cl_fl_cl_ffthuman.h2')
    [pred_y, loss1, loss2] = model_train.predict([X_FFT, X_DPCP, np.ones(len(X))])
    class_y = np.zeros((len(X), 6))
    for i in range(len(X)):
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
    target_names = ['0', '1', '2', '3', '4', '5']
    SP = np.zeros((6, 1))
    SN = np.zeros((6, 1))
    MCC = np.zeros((6, 1))
    Acc = np.zeros((6, 1))
    result = []
    for i, t in enumerate(target_names):
        t_test_y = test_y_oh[:, i]
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
test_model()