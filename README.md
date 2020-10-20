# FR_EN_translation
Neural machine translation


Data available here: https://download.pytorch.org/tutorial/data.zip

Some inspiration from https://github.com/bentrevett/pytorch-seq2seq

Usage:

import data_downloader, model_classes, model_pipeline
french, english, train_data, valid_data, test_data = data_downloader.prepare_data()
gru = model_classes.EncoderDecoderGRU(french, english, 1, 'vanilla', bidirectional_encoder=False)
model_pipeline.train(gru, train_data, valid_data)
gru.plot_losses()
model_pipeline.translate('je viens de manger une pomme', french, english, gru)
