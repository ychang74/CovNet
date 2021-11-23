# BaselineCnnEmbedding conv# parameters
# layer_paras = {'layer': [conv#_in, conv#_out, bn#]}
layer_paras = {'conv1': [1, 64, 128, 128, 256, 256, 256, 512, 512],
               'conv2': [1, 64, 64, 64, 128, 256, 256, 512, 512],
               'conv3': [1, 64, 64, 64, 128, 128, 128, 256, 512],
               'default': [1, 64, 64, 64, 128, 128, 128, 256, 256],
               }

layer_name = {'conv1': ['conv1', 'conv_block1'],
              'conv2': ['conv2', 'conv_block2', 'resblock1', 'mobileblock1'],
              'conv3': ['conv3', 'conv_block3', 'resblock2', 'mobileblock2'],
              }



# Transfer Learning
outer_layer_name = {'conv1': ['conv1', 'conv_block1', 'conv1', 'conv1'],
              'conv2': ['conv2', 'conv_block2', 'resblock1', 'mobileblock1'],
              'conv3': ['conv3', 'conv_block3', 'resblock2', 'mobileblock2'],
              }

inner_layer_name = {'block': ['.conv1.weight', '.conv2.weight', '.bn1.weight', '.bn2.weight', '.bn1.bias', '.bn2.bias'],
                    'res':['.conv3.weight', '.bn3.weight', '.bn3.bias']
                    }


networks = ['baseline', 'vgg', 'resnet', 'mobilenet']
