import os
import sys
import data_loader as dl

from models.fc2_20_2dense import FC2_20_2Dense

from urllib.request import urlopen
import tensorflow as tf
from keras import optimizers as opti
import keras.backend.tensorflow_backend as KTF


def get_sequence(proteinid):
##This function gets protein fasta sequence from uniprot
    link = 'https://www.uniprot.org/uniprot/'+proteinid+'.fasta'          
    myfile = urlopen(link).read().decode('utf-8')
    return ''.join(myfile.split('\n')[1:])

def factory_optimizer( optimizer_name, lr = 0.001 ):
    if optimizer_name == 'sgd':
        return opti.SGD( lr=lr ), 'sgd'
    elif optimizer_name == 'rmsprop':
        return opti.RMSprop( lr=lr ), 'rmsprop'
    elif optimizer_name == 'adagrad':
        return opti.Adagrad( lr=lr ), 'adagrad'
    elif optimizer_name == 'adadelta':
        return opti.Adadelta( lr=lr ), 'adadelta'
    elif optimizer_name == 'adamax':
        return opti.Adamax( lr=lr ), 'adamax'
    elif optimizer_name == 'nadam':
        return opti.Nadam( lr=lr ), 'nadam'
    else:
        return opti.Adam( lr=lr ), 'adam'
def factory_model( model_name ):
    model_name == 'fc2_20_2dense'
    return FC2_20_2Dense(), 'fc2_20_2dense'
    
if __name__ == '__main__':

   #python3 main.py -test ../data/tiny_1166_test.txt -load weights/fc2_20_2dense_2019-01-08_06:32_gpu-0-1_adam_0.001_2048_25_AA24_mirror-medium.h5 -model fc2_20_2dense
    p1=sys.argv[1]
    p2=sys.argv[2]
    remove_polyq=int(sys.argv[3])
    seq_p1=get_sequence(p1)
    seq_p2=get_sequence(p2)
    if remove_polyq==1:
        seq_p1=seq_p1.replace('QQQQQQQQQQQQQQQQQQQQQQQ','')
        seq_p2=seq_p2.replace('QQQQQQQQQQQQQQQQQQQQQQQ','')

    text_file = open("temp_test.txt", "w")
    text_file.write('1166'+'\n'+p1+' '+p2+' '+seq_p1+' '+seq_p2+' '+str(1))
    text_file.close()
    model_name='fc2_20_2dense'
    if 'embed' in model_name:
        test_p1, test_p2, test_labels = dl.load_data_embed( "temp_test.txt" )

    else:
        test_p1, test_p2, test_labels = dl.load_data( "temp_test.txt" )
        test_data = [test_p1, test_p2]
    
    abstract_model, model_name = factory_model( model_name )
    model = abstract_model.get_model()
    optimizer, optimizer_name = factory_optimizer( 'adam', 0.001 )
    model.compile( optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'] )    
    model.load_weights('fc2_20_2dense_model.h5')
    
    score, acc = model.evaluate( test_data, test_labels )

    predict = model.predict( test_data, batch_size=64, verbose=1 )

    print('The interaction probability between '+p1+' and '+p2+' is: ' +str(predict[0][0]) )



        


